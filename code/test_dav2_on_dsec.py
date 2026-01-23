import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events import Tencode, TencodePixelCount
from networks.dav2_wrapper import Dav2
from evaluation import prepare_target_data_torch, prepare_target_data
from losses import normalized_depth_scale_and_shift


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAV2 on a DSEC sequence and visualize depth.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/train/zurich_city_01_e",
        help="Path to a DSEC sequence root (default: datasets/DSEC/data/validate/interlaken_00_c)",
    )
    parser.add_argument(
        "--index", type=int, default=10, help="Index within the sequence to visualize (depth-aligned events)."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl"],
        help="Which DAV2 encoder to use.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to models/dav2/checkpoints/depth_anything_v2_<encoder>.pth",
    )
    parser.add_argument("--input-size", type=int, default=518, help="Square resize fed into the model.")
    parser.add_argument(
        "--time-window-ms",
        type=int,
        default=50,
        help="Event window size for building tencode representations.",
    )
    parser.add_argument("--output-dir", type=str, default="output/test_dav2_on_dsec", help="Where to save visualizations.")
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def to_rgb_events(event_tensor: torch.Tensor) -> np.ndarray:
    """Convert tencode events (3,H,W) to RGB uint8 image."""
    event_np = event_tensor.detach().cpu().numpy()
    event_np = np.transpose(event_np, (1, 2, 0))  # HWC
    event_np = (255 * np.clip(event_np, 0.0, 1.0)).astype(np.uint8)
    return event_np


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    cmap = plt.get_cmap("viridis")
    depth_rgb = (255 * cmap(depth_norm)[..., :3]).astype(np.uint8)
    return depth_rgb


def depth_to_colormap_with_cbar(depth: np.ndarray, path: str) -> None:
    depth_min, depth_max = depth.min(), depth.max()
    fig, ax = plt.subplots()
    im = ax.imshow(depth, cmap='viridis', vmin=depth_min, vmax=depth_max)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Depth (m)')
    ax.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def visualize(sample: dict, pred_np: np.ndarray, pred_np_raw: np.ndarray, target_np: np.ndarray, out_dir: str, idx: int) -> None:
    events = sample["depth_aligned_events"][0]
    events_rgb = to_rgb_events(events)

    pred_rgb = depth_to_colormap(pred_np)
    gt_rgb = depth_to_colormap(target_np)

    # Overlay: blend scaled prediction and ground truth
    overlay_rgb = 0.5 * pred_rgb + 0.5 * gt_rgb

    # Error map: absolute difference, masked to valid GT pixels
    error = np.abs(pred_np - target_np)
    error[target_np == 0] = 0  # mask invalid pixels
    print("Error summary: min {:.4f}, max {:.4f}, mean {:.4f}, median {:.4f}".format(
        error.min(), error.max(), error.mean(), np.median(error[error > 0])
    ))
    error_rgb = depth_to_colormap(error)

    plt.imsave(os.path.join(out_dir, f"{idx:05d}_events.png"), events_rgb)
    depth_to_colormap_with_cbar(pred_np_raw, os.path.join(out_dir, f"{idx:05d}_pred_raw_depth.png"))
    depth_to_colormap_with_cbar(pred_np, os.path.join(out_dir, f"{idx:05d}_pred_scaled_depth.png"))
    depth_to_colormap_with_cbar(target_np, os.path.join(out_dir, f"{idx:05d}_gt_depth.png"))
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_overlay.png"), overlay_rgb.astype(np.uint8))
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_error.png"), error_rgb)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rep = Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=True)
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

    model = Dav2(encoder=args.encoder, checkpoint=args.checkpoint, device=device, input_size=args.input_size)

    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    depth_pred = model(events).squeeze(1)  # [B,H,W]

    # Invert depth prediction if it's in inverse depth
    # normalize per-image (so inversion is well-behaved)
    # pred_min = depth_pred.amin(dim=(1,2), keepdim=True)
    # pred_max = depth_pred.amax(dim=(1,2), keepdim=True)
    # depth_pred = (depth_pred - pred_min) / (pred_max - pred_min + 1e-6)

    # # invert if needed (now near small / far large, or the opposite as you prefer)
    # depth_pred = 1.0 - depth_pred
    # depth_pred = depth_pred * (pred_max - pred_min) + pred_min  # scale back to original range
    # # depth_pred = depth_pred * 8.3576 - 58.7456
    
    # Apply scale-shift normalization to match ground truth
    target_depth_t = sample["depth"][0].to(device)
    target_proc_t = prepare_target_data_torch(target_depth_t, 80.0)
    scale, shift = normalized_depth_scale_and_shift(
        depth_pred, target_proc_t, target_proc_t > 0
    )

    print(scale, shift)
    
    pred_depth_scaled = scale[:, None, None] * depth_pred + shift[:, None, None]
    pred_np = np.clip(pred_depth_scaled.detach().cpu().squeeze().numpy(), 0, 80.0)
    pred_np_raw = np.clip(depth_pred.detach().cpu().squeeze().numpy(), 0, 80.0)
    target_np = prepare_target_data(target_proc_t.detach().cpu().squeeze().numpy(), 80.0)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample, pred_np, pred_np_raw, target_np, out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
