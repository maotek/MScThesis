import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import VoxelGrid
from networks.etnet_dav2 import ETNetDav2
from evaluation import prepare_target_data_torch, prepare_target_data
from losses import normalized_depth_scale_and_shift
from util import save_depth_colormap, save_grayscale, save_voxelgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ET-Net->DAV2 on a DSEC sequence and visualize input, reconstruction, and depth."
    )
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/validate/zurich_city_01_f",
        help="Path to a DSEC sequence root",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=200,
        help="Index within the sequence to visualize (depth-aligned events).",
    )
    parser.add_argument(
        "--time-window-ms",
        type=int,
        default=50,
        help="Event window size for building voxel-grid representations.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins for the voxel grid; must match the ET-Net input.",
    )
    parser.add_argument(
        "--etnet-checkpoint",
        type=str,
        default=None,
        help="Optional ET-Net checkpoint path. Defaults to models/etnet/checkpoints/etnet.pth",
    )
    parser.add_argument(
        "--dav2-checkpoint",
        type=str,
        default=os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth"),
        help="DAV2 checkpoint path (default: models/dav2/checkpoints/depth_anything_v2_vits.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/test_etnet_dav2_on_dsec",
        help="Where to save visualizations.",
    )
    parser.add_argument(
        "--clip-distance",
        type=float,
        default=80.0,
        help="Max depth value (meters) for visualization clipping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Defaults to auto-selection.",
    )
    return parser.parse_args()


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    cmap = plt.get_cmap("viridis")
    depth_rgb = (255 * cmap(depth_norm)[..., :3]).astype(np.uint8)
    return depth_rgb


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def visualize(sample: dict, pred_np: np.ndarray, pred_np_raw: np.ndarray, target_np: np.ndarray, out_dir: str, idx: int) -> None:
    events_voxel = sample["depth_aligned_events"][0]

    pred_raw_rgb = depth_to_colormap(pred_np_raw)
    pred_rgb = depth_to_colormap(pred_np)
    gt_rgb = depth_to_colormap(target_np)

    # Overlay: blend scaled prediction and ground truth
    overlay_rgb = 0.5 * pred_rgb + 0.5 * gt_rgb

    save_voxelgrid(os.path.join(out_dir, f"{idx:05d}_events.png"), events_voxel)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_pred_raw_depth.png"), pred_raw_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_pred_scaled_depth.png"), pred_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_gt_depth.png"), gt_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_overlay.png"), overlay_rgb.astype(np.uint8))


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    rep = VoxelGrid(channels=args.num_bins, height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True)

    dataset = DsecSequence(
        sequence_path=args.sequence,
        event_representation=rep,
        time_window_ms=args.time_window_ms,
        augmentator=None,
        load_images=True,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        split="train",
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    model = ETNetDav2(
        etnet_checkpoint=args.etnet_checkpoint,
        dav2_encoder="vits",
        dav2_checkpoint=args.dav2_checkpoint,
        device=device,
    )

    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)

    # Run ET-Net once to get intensity, then DAV2 for depth
    intensity = model.etnet(events)  # (B,1,H,W) or (B,3,H,W)
    intensity_3ch = intensity.repeat(1, 3, 1, 1) if intensity.shape[1] == 1 else intensity
    depth = model.dav2(intensity_3ch)  # (B,1,H,W)

    # Apply scale-shift normalization to match ground truth
    target_depth_t = sample["depth"][0].to(device)
    target_proc_t = prepare_target_data_torch(target_depth_t, 80.0)
    scale, shift = normalized_depth_scale_and_shift(
        depth.squeeze(1), target_proc_t, target_proc_t > 0
    )
    pred_depth_scaled = scale[:, None, None] * depth.squeeze(1) + shift[:, None, None]
    pred_np = np.clip(pred_depth_scaled.detach().cpu().squeeze().numpy(), 0, 80.0)
    pred_np_raw = np.clip(depth.squeeze().detach().cpu().numpy(), 0, 80.0)
    target_np = prepare_target_data(target_proc_t.detach().cpu().squeeze().numpy(), 80.0)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample, pred_np, pred_np_raw, target_np, out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
