import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

from pprint import pprint

import numpy as np
import torch
import tqdm

from datasets.events.events_representations import E2vidVoxelGrid
from networks.dae import DAE
from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from datasets.events import Tencode, TencodePixelCount, VoxelGrid, ETNetVoxelGrid
from networks.dav2 import Dav2
from networks.e2vid_dav2 import E2VIDDav2
from networks.e2vid_dav2_composite import E2VIDDav2Composite
from networks.etnet_dav2 import ETNetDav2
from evaluation import (
    add_to_metrics,
    prepare_target_data,
    prepare_target_data_torch,
)
from losses import normalized_depth_scale_and_shift
from util import (
    depth_to_colormap,
    save_depth_colormap,
    save_rgb,
    save_voxelgrid,
    voxelgrid_to_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DAV2 on DSEC validation sequences.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (same default as models/depthanyevent/test.py).",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        default=None,
        help="Optional JSON config path; if provided, overrides other args except --csv-path.",
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        type=str,
        default=None,
        help="Optional CSV output path for metrics.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Load config JSON and split into data_loader_config and model_config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    if "data_loader" not in config or "model" not in config:
        raise KeyError("Config must contain top-level 'data_loader' and 'model'")

    data_loader_config = dict(config["data_loader"])
    model_config = dict(config["model"])
    config = {k: v for k, v in config.items() if k not in ("data_loader", "model")}

    return data_loader_config, model_config, config


def setup_device_and_seeds(args: argparse.Namespace) -> torch.device:
    """DepthAnyEvent-style seed init with default seed=42."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") # testing
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    return device


def write_csv(path: str, seq_results: list, mean_metrics: Dict[str, float]) -> None:
    """Write CSV with metrics as rows and sequences as columns plus MEAN."""
    header = ["METRIC", *[r["name"] for r in seq_results], "MEAN"]

    metric_keys = set(mean_metrics.keys())
    for r in seq_results:
        metric_keys.update(r["avg"].keys())

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for k in sorted(metric_keys):
            row = [k]
            for r in seq_results:
                val = r["avg"].get(k)
                row.append("" if val is None else f"{val:.6f}")
            mean_val = mean_metrics.get(k)
            row.append("" if mean_val is None else f"{mean_val:.6f}")
            f.write(",".join(row) + "\n")


def save_visualization(
    seq_name: str,
    idx: int,
    events: torch.Tensor,
    pred_np: np.ndarray,
    vis_dir: str,
) -> None:
    seq_dir = os.path.join(vis_dir, seq_name)
    os.makedirs(seq_dir, exist_ok=True)

    events_chw = events.detach().cpu().squeeze(0)  # (C,H,W)
    events_path = os.path.join(seq_dir, f"{idx:05d}_events.png")
    if events_chw.shape[0] == 3:
        save_rgb(events_path, events_chw)
    else:
        save_voxelgrid(events_path, events_chw)

    save_depth_colormap(os.path.join(seq_dir, f"{idx:05d}_pred.png"), pred_np)


def evaluate_sequence(
    seq_name: str,
    data_loader: torch.utils.data.DataLoader,
    model: object,
    device: torch.device,
    clip_distance: float,
    use_scaleshift: bool,
    representation: str,
    model_name: str,
    vis_interval: int,
    vis_dir: str,
) -> Tuple[Dict[str, float], int]:
    
    model.eval()
    load_images = getattr(data_loader.dataset, "load_images", False)

    metrics_sum: Dict[str, float] = {}
    num_frames = len(data_loader)

    for idx, sample in enumerate(tqdm.tqdm(data_loader, total=num_frames, desc=f"{seq_name}", leave=False)):
        # sample: dict with shapes [B, T, C, H, W], B=1 and T=1 for our settings.
        target_depth_t = sample["depth"][:, 0, 0].to(device)  # (B,H,W)
        events = sample["depth_aligned_events"][:, 0].to(device)  # (B,C,H,W)

        # Use RGB input for dav2_rgb model
        if load_images and representation == "rgb":
            events = sample["rgb"][:, 0].to(device)
            target_hw = target_depth_t.shape[-2:]
            if events.shape[-2:] != target_hw:
                events = torch.nn.functional.interpolate(
                    events,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
        
        pred_depth = model(events)  # (1,1,320,640)
        
        if model_name in ("e2vid_dav2", "etnet_dav2", "dav2_rgb", "dav2", "dav2_composite"):
            pred_depth = 1.0 / (pred_depth + 1)  # convert to depth in ~meters
    
        pred_depth = pred_depth.squeeze(1)  # (1,320,640)
        target_proc_t = prepare_target_data_torch(target_depth_t, clip_distance)

        # Apply scale-shift normalization to match ground truth
        if use_scaleshift:
            scale, shift = normalized_depth_scale_and_shift(
                pred_depth, target_proc_t, target_proc_t > 0
            )
            pred_depth = scale * pred_depth + shift

        pred_np = np.clip(pred_depth.detach().cpu().squeeze().numpy(), 0, clip_distance)
        target_np = target_proc_t.detach().cpu().squeeze().numpy()

        mask = np.ones_like(target_np, dtype=bool)

        metrics_sum = add_to_metrics(
            idx,
            metrics_sum,
            target_np,
            pred_np,
            mask,
            event_frame=None,
            prefix="_",
            debug=False,
            output_folder=None,
        )

        # Visualization of predictions at intervals
        if vis_interval > 0 and idx % vis_interval == 0:
            save_visualization(
                seq_name=seq_name,
                idx=idx,
                events=events,
                pred_np=pred_depth.detach().cpu().squeeze().numpy(),
                vis_dir=vis_dir,
            )

        for depth_threshold in (10, 20, 30):
            threshold_mask = np.nan_to_num(target_np) < depth_threshold
            combined_mask = mask & threshold_mask
            metrics_sum = add_to_metrics(
                -1,
                metrics_sum,
                target_np,
                pred_np,
                combined_mask,
                event_frame=None,
                prefix=f"_{depth_threshold}_",
                debug=False,
                output_folder=None,
            )

    return metrics_sum, num_frames


def accumulate_metrics(target: Dict[str, float], source: Dict[str, float]) -> Dict[str, float]:
    for k, v in source.items():
        target[k] = target.get(k, 0.0) + v
    return target



def fetch_model(model_config: Dict[str, object], device: torch.device, representation: str = "") -> object:
    model_name = str(model_config["model_type"])

    if model_name == "dav2":
        return Dav2(
            encoder=str(model_config.get("encoder", "vits")),
            checkpoint=model_config.get("checkpoint", os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth")),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            rgb=(representation.lower() == "rgb"),
            device=device,
        )
    elif model_name == "e2vid_dav2":
        return E2VIDDav2(
            e2vid_weights=model_config.get("e2vid_weights", os.path.join("models", "rpg_e2vid", "pretrained", "E2VID_lightweight.pth.tar")),
            dav2_encoder=str(model_config.get("dav2_encoder", "vits")),
            dav2_checkpoint=model_config.get("dav2_checkpoint", os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth")),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            device=device,
        )
    elif model_name == "e2vid_dav2_composite":
        return E2VIDDav2Composite(
            e2vid_weights=model_config.get("e2vid_weights", os.path.join("models", "rpg_e2vid", "pretrained", "E2VID_lightweight.pth.tar")),
            dav2_encoder=str(model_config.get("dav2_encoder", "vits")),
            dav2_checkpoint=model_config.get("dav2_checkpoint", os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth")),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            device=device,
        )
    elif model_name == "dae":
        return DAE(
            checkpoint=model_config.get("checkpoint", os.path.join("models", "depthanyevent", "checkpoints", "finetuned_dsec.pth")),
            input_channels=int(model_config.get("input_channels", 3)),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            inv_prediction=bool(model_config.get("inv_prediction", True)),
            activation=str(model_config.get("activation", "relu")),
            scale_factor=float(model_config.get("scale_factor", 1.0)),
            nopretrain=bool(model_config.get("nopretrain", False)),
            device=device,
        )
    elif model_name == "etnet_dav2":
        return ETNetDav2(
            etnet_checkpoint=model_config.get("etnet_checkpoint", os.path.join("models", "etnet", "checkpoints", "etnet.pth")),
            dav2_encoder=str(model_config.get("dav2_encoder", "vits")),
            dav2_checkpoint=model_config.get("dav2_checkpoint", os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth")),
            input_size_width=int(model_config.get("input_size_width", 350)),
            input_size_height=int(model_config.get("input_size_height", 266)),
            device=device,
        )
    else:
        print(model_name)
        raise ValueError(f"Unsupported model: {model_name}")



def main() -> None:
    args = parse_args()

    # Setting up device and seeds
    print("Setting up device and seeds...")
    device = setup_device_and_seeds(args)

    # Ensure CSV_path directory exists
    print("Preparing CSV output directory...")
    if args.csv_path:
        csv_dir = os.path.dirname(os.path.abspath(args.csv_path))
        os.makedirs(csv_dir, exist_ok=True)

    # Reading config
    print("Loading configuration...")
    data_loader_config, model_config, config = load_config(args.config_path)

    # Ensure visualization directory exists
    vis_dir = config.get("vis_dir", "visualizations")
    if config.get("vis_interval", 0) > 0:
        os.makedirs(vis_dir, exist_ok=True)

    representation = data_loader_config.get("event_representation", {}).get("representation_type", "")

    model = fetch_model(model_config, device, representation=representation)

    metrics_sequence_dict: Dict[str, Dict[str, float]] = {}
    seq_results = []
    data_loaders_dict = fetch_dsec_dataloader(data_loader_config, test=True)

    for seq_name, data_loader in data_loaders_dict.items():
        metrics_sum, frames = evaluate_sequence(
            seq_name=seq_name,
            data_loader=data_loader,
            model=model,
            device=device,
            clip_distance=config.get("clip_distance", 80.0),
            use_scaleshift=config.get("use_scaleshift", True),
            representation=representation,
            model_name=model_config["model_type"],
            vis_interval=config.get("vis_interval", 0),
            vis_dir=vis_dir,
        )

        seq_avg = {k: v / frames for k, v in metrics_sum.items()}
        print(f"\nSequence {seq_name} ({frames} frames):")
        for k in sorted(seq_avg.keys()):
            print(f"  {k}: {seq_avg[k]:.6f}")

        metrics_sequence_dict[seq_name] = seq_avg
        seq_results.append({"name": seq_name, "frames": frames, "avg": seq_avg})

    # Overall average metrics across sequences
    metrics_mean: Dict[str, list] = {}
    for seq in metrics_sequence_dict:
        for k in metrics_sequence_dict[seq]:
            if k not in metrics_mean:
                metrics_mean[k] = []
            metrics_mean[k].append(np.nanmean(np.array(metrics_sequence_dict[seq][k])))

    if len(metrics_mean) > 0:
        overall_avg = {k: float(np.nanmean(np.array(metrics_mean[k]))) for k in metrics_mean}
        print("\n================ Overall (validation) ===============")
        print(f"Sequences: {len(metrics_sequence_dict)}")
        for k in sorted(overall_avg.keys()):
            print(f"{k}: {overall_avg[k]:.6f}")

        if args.csv_path:
            write_csv(args.csv_path, seq_results, overall_avg)




if __name__ == "__main__":
    main()
