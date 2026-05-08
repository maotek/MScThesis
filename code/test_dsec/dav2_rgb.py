import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
from pprint import pprint
import torch
import matplotlib.pyplot as plt

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from networks.dav2 import Dav2
from datasets.events import Tencode
from datasets.utils import fetch_preprocessing
from evaluation import add_to_metrics, prepare_target_data_torch
from losses import normalized_depth_scale_and_shift
from util import (
    depth_to_colormap,
    rgb_to_uint8,
    save_depth_colormap_with_cbar,
    save_image,
    save_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAV2 on RGB images from a DSEC sequence and visualize depth.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/validation/interlaken_00_f",
        help="Path to a DSEC sequence root",
    )
    parser.add_argument(
        "--index", type=int, default=50, help="Index within the sequence to visualize."
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
    parser.add_argument("--output-dir", type=str, default="test_dsec_output/test_dav2_rgb_on_dsec", help="Where to save visualizations.")
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def visualize(sample: dict, pred_np: np.ndarray, pred_np_raw: np.ndarray, target_np: np.ndarray, out_dir: str, idx: int) -> None:
    rgb = sample["rgb"][0]
    input_rgb = rgb

    pred_rgb = depth_to_colormap(pred_np)
    gt_rgb = depth_to_colormap(target_np)

    # Overlay: blend scaled prediction and ground truth
    overlay_rgb = 0.5 * pred_rgb + 0.5 * gt_rgb

    # Error map: absolute difference, masked to valid GT pixels
    error = np.abs(pred_np - target_np)
    error[target_np == 0] = 0  # mask invalid pixels
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

    input_uint8 = rgb_to_uint8(input_rgb)
    grid_items = [
        ("RGB", input_uint8, None),
        ("Pred (raw)", pred_np_raw, ("viridis", pred_raw_min, pred_raw_max)),
        ("Pred (scaled)", pred_np, ("viridis", pred_min, pred_max)),
        ("GT", target_np, ("viridis", gt_min, gt_max)),
        ("Overlay", overlay_rgb.astype(np.uint8), None),
        ("Error", error, ("magma", 0.0, error_vmax)),
    ]

    save_rgb(os.path.join(out_dir, f"{idx:05d}_input.png"), input_rgb)
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_pred_raw_depth.png"), pred_np_raw)
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_pred_scaled_depth.png"), pred_np)
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_gt_depth.png"), target_np)
    save_image(os.path.join(out_dir, f"{idx:05d}_overlay.png"), overlay_rgb.astype(np.uint8))
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_error.png"), error_rgb)
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
        event_representation=Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=False),  # dummy rep
        time_window_ms=50,
        augmentator=augmentator,
        load_images=True,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            "models", "dav2", "checkpoints", f"depth_anything_v2_{args.encoder}.pth"
        )
    model = Dav2(
        encoder=args.encoder,
        checkpoint=checkpoint_path,
        device=device,
        input_size_height=266,
        input_size_width=350,
        normalize_imagenet=True,
    )

    rgb = sample["rgb"][0].unsqueeze(0).to(device)
    rgb = torch.nn.functional.interpolate(rgb, size=(320, 640), mode='bilinear', align_corners=False)
    print("min/max RGB input:", rgb.min().item(), rgb.max().item())
    depth_pred = model(rgb)
    # torch.save(depth_pred, os.path.join(args.output_dir, f"{args.index:05d}_rgb_tensor.pt"))

    # depth_pred = torch.load(os.path.join(args.output_dir, f"{args.index:05d}_rgb_tensor.pt"))
    depth_pred = 1.0 / (depth_pred + 2) # Convert from inverse depth to depth in meters

    # Apply scale-shift normalization to match ground truth
    target_depth_t = sample["depth"][0].to(device)
    target_proc_t = prepare_target_data_torch(target_depth_t, 80.0)
    scale, shift = normalized_depth_scale_and_shift(
        depth_pred.squeeze(1), target_proc_t, target_proc_t > 0
    )
    pred_depth_scaled = scale * depth_pred + shift
    pred_np = np.clip(pred_depth_scaled.detach().cpu().squeeze().numpy(), 0, 80.0)
    pred_np_raw = depth_pred.detach().cpu().squeeze().numpy()
    target_np = target_proc_t.detach().cpu().squeeze().numpy()

    mask = np.ones_like(target_np, dtype=bool)

    metrics_sum = add_to_metrics(
        0,
        {},
        target_np,
        pred_np,
        mask,
        event_frame=None,
        prefix="_",
        debug=False,
        output_folder=None,
    )

    metrics_filtered = {
        k: v
        for k, v in metrics_sum.items()
        if not k.startswith(("_10_", "_20_", "_30_"))
    }
    pprint(metrics_filtered)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample, pred_np, pred_np_raw, target_np, out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
