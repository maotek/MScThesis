import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import E2DepthVoxelGrid, VoxelGrid
from networks.e2depth import load_e2depth
from datasets.utils import fetch_preprocessing
from evaluation import prepare_target_data_torch
from losses import normalized_depth_scale_and_shift
from util import depth_to_colormap, voxelgrid_to_uint8, save_image, save_voxelgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2Depth on a DSEC sequence and visualize depth prediction.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/validation/interlaken_00_f",
        help="Path to a DSEC sequence root",
    )
    parser.add_argument("--index", type=int, default=50, help="Index within the sequence to visualize (depth-aligned events).")
    parser.add_argument("--time-window-ms", type=int, default=50, help="Event window size for building voxel-grid representations.")
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins for the voxel grid; must match the checkpoint's num_bins (E2Depth default uses 5).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional E2Depth checkpoint path. Defaults to models/rpg_e2depth/pretrained/E2DEPTH_si_grad_loss_mixed.pth.tar",
    )
    parser.add_argument("--output-dir", type=str, default="test_dsec_output/test_e2depth_on_dsec", help="Where to save visualizations.")
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def visualize(sample: dict, pred_np: np.ndarray, pred_np_raw: np.ndarray, target_np: np.ndarray, out_dir: str, idx: int) -> None:
    # events_voxel: (C,H,W), depth: (1,H,W) in log-depth encoding
    events_voxel = sample["depth_aligned_events"][0]

    save_voxelgrid(os.path.join(out_dir, f"{idx:05d}_events.png"), events_voxel)

    pred_raw_rgb = depth_to_colormap(pred_np_raw)
    pred_scaled_rgb = depth_to_colormap(pred_np)
    gt_rgb = depth_to_colormap(target_np)

    # Overlay: blend scaled prediction and ground truth
    overlay_rgb = 0.5 * pred_scaled_rgb + 0.5 * gt_rgb

    # Error map: absolute difference, masked to valid GT pixels
    error = np.abs(pred_np - target_np)
    error[target_np == 0] = 0
    error_rgb = depth_to_colormap(error)

    def depth_limits(arr: np.ndarray, fallback=(0.0, 80.0)) -> tuple[float, float]:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmax - vmin < 1e-6:
            return fallback
        return vmin, vmax

    pred_raw_min, pred_raw_max = depth_limits(pred_np_raw)
    pred_min, pred_max = depth_limits(pred_np)
    gt_min, gt_max = depth_limits(target_np)
    error_vmax = max(float(error.max()), 1e-6)

    events_uint8 = voxelgrid_to_uint8(events_voxel)
    grid_items = [
        ("Events", events_uint8, None),
        ("Pred (raw)", pred_np_raw, ("viridis", pred_raw_min, pred_raw_max)),
        ("Pred (scaled)", pred_np, ("viridis", pred_min, pred_max)),
        ("GT", target_np, ("viridis", gt_min, gt_max)),
        ("Overlay", overlay_rgb.astype(np.uint8), None),
        ("Error", error, ("magma", 0.0, error_vmax)),
    ]

    save_image(os.path.join(out_dir, f"{idx:05d}_pred_raw_depth.png"), pred_raw_rgb)
    save_image(os.path.join(out_dir, f"{idx:05d}_pred_scaled_depth.png"), pred_scaled_rgb)
    save_image(os.path.join(out_dir, f"{idx:05d}_gt_depth.png"), gt_rgb)
    save_image(os.path.join(out_dir, f"{idx:05d}_overlay.png"), overlay_rgb.astype(np.uint8))
    save_image(os.path.join(out_dir, f"{idx:05d}_error.png"), error_rgb)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (title, img, cmap_cfg) in zip(axes.flat, grid_items):
        if cmap_cfg is None:
            im = ax.imshow(img)
        else:
            cmap, vmin, vmax = cmap_cfg
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{idx:05d}_grid.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rep = E2DepthVoxelGrid(channels=args.num_bins, height=DSEC_HEIGHT, width=DSEC_WIDTH)
    preprocess_config = [
        {
            "preprocessing_type": "CenterCrop",
            "height": 320,
            "width": 640,
        }
    ]
    augmentator = fetch_preprocessing(preprocess_config)
    dataset = DsecSequence(
        sequence_path=args.sequence,
        event_representation=rep,
        time_window_ms=args.time_window_ms,
        augmentator=augmentator,
        load_images=False,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    model = load_e2depth(weights_path=args.checkpoint, device=device)
    # E2Depth expects (B,C,H,W)
    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    depth = model(events)  # (B,1,H,W)

    target_depth_t = sample["depth"][0].to(device)
    target_proc_t = prepare_target_data_torch(target_depth_t, 80.0)

    depth_log = depth.squeeze(1)
    pred_raw = torch.exp(3.70378 * (depth_log - 1.0))
    scale, shift = normalized_depth_scale_and_shift(
        pred_raw, target_proc_t, target_proc_t > 0
    )
    pred_scaled = scale * pred_raw + shift

    pred_np = np.clip(pred_scaled.detach().cpu().squeeze().numpy(), 0, 80.0)
    pred_np_raw = pred_raw.detach().cpu().squeeze().numpy()
    target_np = target_proc_t.detach().cpu().squeeze().numpy()

    out_dir = ensure_dir(args.output_dir)
    visualize(sample, pred_np, pred_np_raw, target_np, out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
