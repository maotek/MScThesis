import argparse
import os
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from evaluation import add_to_metrics, prepare_target_data_torch
from losses import normalized_depth_scale_and_shift
from networks.dae import DAE
from networks.fully_conv import FullyConv
from networks.dav2 import Dav2
from util import rgb_to_uint8, voxelgrid_to_uint8

BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FullyConv->DAv2 (FullyConv weights only) against DAE on the full DSEC "
            "validation split, with side-by-side visualizations."
        )
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Optional JSON config (same format as evaluate.py) to override DAE/data settings.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="datasets/DSEC/data",
        help="Root path to DSEC dataset (contains train/validation/test folders).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Which DSEC split to evaluate (train/validation/test).",
    )
    parser.add_argument(
        "--time-window-ms",
        type=int,
        default=50,
        help="Event window size for building event representations.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins/channels for voxel grid (FullyConv input).",
    )
    parser.add_argument(
        "--fully-conv-checkpoint",
        type=str,
        default="train_output/train_dsec_fully_conv_dav2_batch10_RC/epoch_050.pt",
        help="Checkpoint containing trained FullyConv weights.",
    )
    parser.add_argument(
        "--dav2-checkpoint",
        type=str,
        default=os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth"),
        help="Pretrained DAv2 checkpoint.",
    )
    parser.add_argument(
        "--dae-checkpoint",
        type=str,
        default=os.path.join(
            "models",
            "depthanyevent",
            "weights",
            "dav2",
            "finetuned_dsec",
            "finetuned_dsec.pth",
        ),
        help="DAE pretrained checkpoint (DSEC finetuned).",
    )
    parser.add_argument(
        "--dae-encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="DepthAnyEvent encoder variant.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_dsec_output/test_fully_conv_vs_dae",
        help="Where to save visualizations.",
    )
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=100,
        help="Save a visualization every N frames (per sequence).",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=100,
        help="Write input/feature stats every N frames (0 = disabled).",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="",
        help="Optional directory for stats JSONL (default: <output-dir>).",
    )
    parser.add_argument(
        "--clip-distance",
        type=float,
        default=80.0,
        help="Max depth value (meters) for clipping and visualization.",
    )
    parser.add_argument(
        "--inv-prediction-constant",
        type=float,
        default=1.0,
        help="Constant used when inverting FullyConv/DAv2 inverse depth outputs.",
    )
    parser.add_argument(
        "--sequence-window",
        type=int,
        default=1,
        help="Temporal window size for DsecSequence.",
    )
    parser.add_argument(
        "--sequence-step",
        type=int,
        default=1,
        help="Temporal step between samples for DsecSequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Optional device override (cuda, mps, cpu).",
    )
    parser.add_argument(
        "--no-inv-prediction",
        dest="inv_prediction",
        action="store_false",
        help="Disable inverse-depth conversion for FullyConv/DAv2 outputs.",
    )
    parser.add_argument(
        "--no-scaleshift",
        dest="use_scaleshift",
        action="store_false",
        help="Disable scale-shift normalization to GT.",
    )
    parser.set_defaults(inv_prediction=True, use_scaleshift=True)
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_fully_conv_model(args: argparse.Namespace, device: torch.device) -> FullyConv:
    model = FullyConv(in_channels=args.num_bins).to(device)

    ckpt = torch.load(args.fully_conv_checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    fc_state = {
        k.replace("fully_conv.", ""): v for k, v in state.items() if k.startswith("fully_conv.")
    }
    if not fc_state:
        raise RuntimeError(
            f"No FullyConv weights found in checkpoint: {args.fully_conv_checkpoint}"
        )
    model.load_state_dict(fc_state, strict=True)
    model.eval()
    return model


def load_dav2_model(args: argparse.Namespace, device: torch.device) -> Dav2:
    model = Dav2(
        encoder="vits",
        checkpoint=args.dav2_checkpoint,
        device=device,
        input_size_width=350,
        input_size_height=266,
        normalize_imagenet=False,
    )
    model.eval()
    return model


def load_dae_model(args: argparse.Namespace, device: torch.device) -> DAE:
    model = DAE(
        encoder=args.dae_encoder,
        checkpoint=args.dae_checkpoint,
        device=device,
        input_size_width=350,
        input_size_height=266,
        activation="relu",
        scale_factor=1.0,
        inv_prediction=True,
        input_channels=3,
        nopretrain=False,
    )
    model.eval()
    return model


def visualize_sample(
    out_dir: str,
    seq_name: str,
    idx: int,
    voxel_events: torch.Tensor,
    tencode_events: torch.Tensor,
    fc_recon: torch.Tensor,
    fc_pred_preinv: np.ndarray,
    fc_pred_raw: np.ndarray,
    fc_pred: np.ndarray,
    dae_pred_raw: np.ndarray,
    dae_pred: np.ndarray,
    target: np.ndarray,
    clip_distance: float,
) -> None:
    seq_dir = os.path.join(out_dir, seq_name)
    os.makedirs(seq_dir, exist_ok=True)

    events_voxel_img = voxelgrid_to_uint8(voxel_events)

    recon = torch.clamp(fc_recon, 0.0, 1.0)
    recon_img = rgb_to_uint8(recon)

    tencode_img = rgb_to_uint8(torch.clamp(tencode_events, 0.0, 1.0))

    fc_error = np.abs(fc_pred - target)
    fc_error[target == 0] = 0
    dae_error = np.abs(dae_pred - target)
    dae_error[target == 0] = 0
    error_vmax = max(float(fc_error.max()), float(dae_error.max()), 1e-6)
    fc_error_sum = float(fc_error.sum())
    dae_error_sum = float(dae_error.sum())

    def depth_limits(arr: np.ndarray) -> tuple[float, float]:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmax - vmin < 1e-6:
            return vmin, vmin + 1e-6
        return vmin, vmax

    fc_preinv_min, fc_preinv_max = depth_limits(fc_pred_preinv)
    fc_raw_min, fc_raw_max = depth_limits(fc_pred_raw)
    dae_raw_min, dae_raw_max = depth_limits(dae_pred_raw)

    grid_items = [
        ("Voxel Events", events_voxel_img, ("gray", 0.0, 255.0)),
        ("FullyConv Recon", recon_img, None),
        (f"FC Pred (pre-inv) min={fc_preinv_min:.2f} max={fc_preinv_max:.2f}", fc_pred_preinv, ("viridis", fc_preinv_min, fc_preinv_max)),
        (f"FC Pred (post-inv) min={fc_raw_min:.2f} max={fc_raw_max:.2f}", fc_pred_raw, ("viridis", fc_raw_min, fc_raw_max)),
        (f"FC Pred (scaled) min={0.0:.2f} max={clip_distance:.2f}", fc_pred, ("viridis", 0.0, clip_distance)),
        (f"GT min={0.0:.2f} max={clip_distance:.2f}", target, ("viridis", 0.0, clip_distance)),
        ("Tencode Events", tencode_img, None),
        (f"DAE Pred min={dae_raw_min:.2f} max={dae_raw_max:.2f}", dae_pred_raw, ("viridis", dae_raw_min, dae_raw_max)),
        (f"DAE Pred (scaled) min={0.0:.2f} max={clip_distance:.2f}", dae_pred, ("viridis", 0.0, clip_distance)),
        (f"FC Error (sum={fc_error_sum:.2f})", fc_error, ("magma", 0.0, error_vmax)),
        (f"DAE Error (sum={dae_error_sum:.2f})", dae_error, ("magma", 0.0, error_vmax)),
    ]

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    for ax, (title, img, cmap_cfg) in zip(axes.flat, grid_items):
        if cmap_cfg is None:
            ax.imshow(img)
        else:
            cmap, vmin, vmax = cmap_cfg
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(seq_dir, f"{idx:05d}_grid.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _tensor_stats(arr) -> dict:
    if isinstance(arr, torch.Tensor):
        arr_np = arr.detach().float().cpu().numpy()
    else:
        arr_np = np.asarray(arr, dtype=np.float32)
    stats = {
        "shape": list(arr_np.shape),
        "min": float(np.nanmin(arr_np)),
        "max": float(np.nanmax(arr_np)),
        "mean": float(np.nanmean(arr_np)),
        "std": float(np.nanstd(arr_np)),
    }
    if arr_np.ndim == 3 and arr_np.shape[0] in (3, 5):
        ch_stats = {}
        for ci in range(arr_np.shape[0]):
            ch = arr_np[ci]
            ch_stats[f"ch{ci}"] = {
                "min": float(np.nanmin(ch)),
                "max": float(np.nanmax(ch)),
                "mean": float(np.nanmean(ch)),
                "std": float(np.nanstd(ch)),
            }
        stats["per_channel"] = ch_stats
    return stats


def _dav2_preprocess(dav2_model: Dav2, x: torch.Tensor) -> torch.Tensor:
    # Mirrors Dav2.infer_image_torch preprocessing.
    orig_hw = x.shape[-2:]
    h, w = orig_hw
    scale = max(dav2_model.input_size_height / float(h), dav2_model.input_size_width / float(w))
    resized_h = int(np.ceil((h * scale) / 14.0) * 14)
    resized_w = int(np.ceil((w * scale) / 14.0) * 14)
    x = torch.nn.functional.interpolate(x, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    if dav2_model.rgb:
        x = (x - dav2_model.imagenet_mean) / dav2_model.imagenet_std
    return x


def _dav2_feature_stats(dav2_model: Dav2, x: torch.Tensor) -> dict:
    # Get intermediate DINOv2 features (reshaped to B,C,H,W) and summarize.
    x_prep = _dav2_preprocess(dav2_model, x)
    layer_idx = dav2_model.model.intermediate_layer_idx[dav2_model.model.encoder]
    feats = dav2_model.model.pretrained.get_intermediate_layers(
        x_prep, layer_idx, reshape=True, return_class_token=False
    )
    stats = {}
    for i, feat in enumerate(feats):
        stats[f"layer_{layer_idx[i]}"] = _tensor_stats(feat)
    return stats


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def update_metrics(
    metrics: dict,
    target: np.ndarray,
    pred: np.ndarray,
) -> dict:
    mask = np.ones_like(target, dtype=bool)
    return add_to_metrics(0, metrics, target, pred, mask, prefix="_")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    args.datapath = resolve_path(args.datapath)
    args.fully_conv_checkpoint = resolve_path(args.fully_conv_checkpoint)
    args.dav2_checkpoint = resolve_path(args.dav2_checkpoint)
    args.dae_checkpoint = resolve_path(args.dae_checkpoint)
    args.output_dir = resolve_path(args.output_dir)
    stats_dir = resolve_path(args.stats_dir) if args.stats_dir else args.output_dir

    # Hardcode DSEC dataloader settings to match
    # code/configs/dsec/validation/dae_tencode_DSEC_checkpoint.json
    args.datapath = resolve_path("datasets/DSEC/data")
    args.split = "validation"
    args.time_window_ms = 50
    args.sequence_window = 1
    args.sequence_step = 1

    out_dir = ensure_dir(args.output_dir)

    # Build models
    fc_model = load_fully_conv_model(args, device)
    dav2_model = load_dav2_model(args, device)
    dae_model = load_dae_model(args, device)

    preprocess_config = [
        {"preprocessing_type": "CenterCrop", "height": 320, "width": 640}
    ]

    data_loader_config_dae = {
        "dataset": "dsec",
        "datapath": args.datapath,
        "split": "validation",
        "concatenate_sequences": False,
        "event_representation": {
            "representation_type": "tencode",
            "normalize": True,
            "white_frame": False,
            "height": DSEC_HEIGHT,
            "width": DSEC_WIDTH,
        },
        "preprocessing": preprocess_config,
        "load_images": False,
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": False,
        "sequence_window": 1,
        "sequence_step": 1,
        "time_window_ms": 50,
    }

    data_loader_config_fc = {
        "dataset": "dsec",
        "datapath": args.datapath,
        "split": "validation",
        "concatenate_sequences": False,
        "event_representation": {
            "representation_type": "voxelgrid",
            "channels": args.num_bins,
            "normalize": True,
            "height": DSEC_HEIGHT,
            "width": DSEC_WIDTH,
        },
        "preprocessing": preprocess_config,
        "load_images": False,
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": False,
        "sequence_window": 1,
        "sequence_step": 1,
        "time_window_ms": 50,
    }

    data_loaders_dae = fetch_dsec_dataloader(data_loader_config_dae, test=True)
    data_loaders_fc = fetch_dsec_dataloader(data_loader_config_fc, test=True)

    seq_names = sorted(set(data_loaders_dae.keys()) & set(data_loaders_fc.keys()))
    if not seq_names:
        raise FileNotFoundError("No sequences found in dataloaders.")
    print(f"Loaded {len(seq_names)} sequences from dataloaders.")

    seq_metrics_fc = []
    seq_metrics_dae = []

    for seq_name in seq_names:
        data_loader_fc = data_loaders_fc[seq_name]
        data_loader_dae = data_loaders_dae[seq_name]

        if len(data_loader_fc) != len(data_loader_dae):
            raise RuntimeError(
                f"Dataloader length mismatch for {seq_name}: {len(data_loader_fc)} vs {len(data_loader_dae)}"
            )

        metrics_fc_seq: dict = {}
        metrics_dae_seq: dict = {}
        count_seq = 0

        for idx, (sample_fc, sample_dae) in enumerate(
            tqdm.tqdm(
                zip(data_loader_fc, data_loader_dae),
                total=len(data_loader_dae),
                desc=f"{seq_name}",
                leave=False,
            )
        ):
            if sample_fc["depth"].shape[0] != 1 or sample_dae["depth"].shape[0] != 1:
                raise RuntimeError("This script expects batch_size=1 in the DSEC dataloaders.")

            target_depth_t = sample_fc["depth"][:, 0, 0].to(device)
            events_fc = sample_fc["depth_aligned_events"][:, 0].to(device)
            events_dae = sample_dae["depth_aligned_events"][:, 0].to(device)

            target_proc_t = prepare_target_data_torch(target_depth_t, args.clip_distance)

            # FullyConv -> DAv2
            fc_recon = fc_model(events_fc)
            fc_depth_preinv = dav2_model(fc_recon)
            fc_depth = fc_depth_preinv
            if args.inv_prediction:
                fc_depth = 1.0 / (fc_depth + args.inv_prediction_constant)

            fc_depth_raw = fc_depth.squeeze(1)
            if args.use_scaleshift:
                scale, shift = normalized_depth_scale_and_shift(
                    fc_depth_raw, target_proc_t, target_proc_t > 0
                )
                fc_depth_scaled = scale * fc_depth + shift
            else:
                fc_depth_scaled = fc_depth

            # DAE
            dae_depth = dae_model(events_dae)
            dae_depth_raw = dae_depth.squeeze(1)
            if args.use_scaleshift:
                scale, shift = normalized_depth_scale_and_shift(
                    dae_depth_raw, target_proc_t, target_proc_t > 0
                )
                dae_depth_scaled = scale * dae_depth + shift
            else:
                dae_depth_scaled = dae_depth

            fc_preinv_np = fc_depth_preinv.detach().cpu().squeeze().numpy()
            fc_raw_np = fc_depth_raw.detach().cpu().squeeze().numpy()
            fc_np = np.clip(fc_depth_scaled.detach().cpu().squeeze().numpy(), 0, args.clip_distance)
            dae_raw_np = dae_depth_raw.detach().cpu().squeeze().numpy()
            dae_np = np.clip(dae_depth_scaled.detach().cpu().squeeze().numpy(), 0, args.clip_distance)
            target_np = target_proc_t.detach().cpu().squeeze().numpy()

            metrics_fc_seq = update_metrics(metrics_fc_seq, target_np, fc_np)
            metrics_dae_seq = update_metrics(metrics_dae_seq, target_np, dae_np)

            if args.stats_interval > 0 and idx % args.stats_interval == 0:
                stats_record = {
                    "sequence": seq_name,
                    "index": idx,
                    "fc_input_stats": _tensor_stats(fc_recon[0]),
                    "dae_input_stats": _tensor_stats(events_dae[0]),
                    "dav2_features_fc": _dav2_feature_stats(dav2_model, fc_recon),
                    "dav2_features_tencode": _dav2_feature_stats(dav2_model, events_dae),
                }
                _append_jsonl(os.path.join(stats_dir, f"{seq_name}.jsonl"), stats_record)

            if args.vis_interval > 0 and idx % args.vis_interval == 0:
                visualize_sample(
                    out_dir=out_dir,
                    seq_name=seq_name,
                    idx=idx,
                    voxel_events=sample_fc["depth_aligned_events"][0, 0],
                    tencode_events=sample_dae["depth_aligned_events"][0, 0],
                    fc_recon=fc_recon[0].detach().cpu(),
                    fc_pred_preinv=fc_preinv_np,
                    fc_pred_raw=fc_raw_np,
                    fc_pred=fc_np,
                    dae_pred_raw=dae_raw_np,
                    dae_pred=dae_np,
                    target=target_np,
                    clip_distance=args.clip_distance,
                )

            count_seq += 1

        if count_seq == 0:
            continue

        avg_fc = {k: v / count_seq for k, v in metrics_fc_seq.items()}
        avg_dae = {k: v / count_seq for k, v in metrics_dae_seq.items()}
        print(f"\nSequence {seq_name} (frames: {count_seq})")
        print("FullyConv->DAv2 metrics:")
        pprint(avg_fc)
        print("DAE metrics:")
        pprint(avg_dae)

        seq_metrics_fc.append(avg_fc)
        seq_metrics_dae.append(avg_dae)


    def mean_over_sequences(seq_metrics: list[dict]) -> dict:
        if not seq_metrics:
            return {}
        keys = set()
        for m in seq_metrics:
            keys.update(m.keys())
        mean_metrics = {}
        for k in keys:
            vals = [m[k] for m in seq_metrics if k in m]
            if vals:
                mean_metrics[k] = float(np.nanmean(np.array(vals)))
        return mean_metrics

    if seq_metrics_fc or seq_metrics_dae:
        avg_fc_seq = mean_over_sequences(seq_metrics_fc)
        avg_dae_seq = mean_over_sequences(seq_metrics_dae)
        print(f"\nOverall (mean over sequences: {len(seq_metrics_dae)})")
        print("FullyConv->DAv2 metrics:")
        pprint(avg_fc_seq)
        print("DAE metrics:")
        pprint(avg_dae_seq)

    print(f"Visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
