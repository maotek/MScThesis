import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from networks.dav2_wrapper import Dav2
from datasets.events import Tencode
from evaluation import prepare_target_data_torch, prepare_target_data
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
        default="datasets/DSEC/data/validate/interlaken_00_c",
        help="Path to a DSEC sequence root (default: datasets/DSEC/data/validate/interlaken_00_c)",
    )
    parser.add_argument(
        "--index", type=int, default=10, help="Index within the sequence to visualize."
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
    parser.add_argument("--output-dir", type=str, default="output/test_dav2_rgb_on_dsec", help="Where to save visualizations.")
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

    save_rgb(os.path.join(out_dir, f"{idx:05d}_input.png"), input_rgb)
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_pred_raw_depth.png"), pred_np_raw)
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_pred_scaled_depth.png"), pred_np)
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_gt_depth.png"), target_np)
    save_image(os.path.join(out_dir, f"{idx:05d}_overlay.png"), overlay_rgb.astype(np.uint8))
    save_depth_colormap_with_cbar(os.path.join(out_dir, f"{idx:05d}_error.png"), error_rgb)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DsecSequence(
        sequence_path=args.sequence,
        event_representation=Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=False),  # dummy rep
        time_window_ms=50,
        augmentator=None,
        load_images=True,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        split="validate",
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    model = Dav2(encoder=args.encoder, checkpoint=args.checkpoint, device=device, input_size=args.input_size)

    rgb = sample["rgb"][0].unsqueeze(0).to(device)
    rgb = torch.nn.functional.interpolate(rgb, size=(480, 640), mode='bilinear', align_corners=False)
    depth_pred = model(rgb)

    # Apply scale-shift normalization to match ground truth
    target_depth_t = sample["depth"][0].to(device)
    target_proc_t = prepare_target_data_torch(target_depth_t, 80.0)
    scale, shift = normalized_depth_scale_and_shift(
        depth_pred.squeeze(1), target_proc_t, target_proc_t > 0
    )
    pred_depth_scaled = scale[:, None, None] * depth_pred.squeeze(1) + shift[:, None, None]
    pred_np = np.clip(pred_depth_scaled.detach().cpu().squeeze().numpy(), 0, 80.0)
    pred_np_raw = np.clip(depth_pred.squeeze().detach().cpu().numpy(), 0, 80.0)
    target_np = prepare_target_data(target_proc_t.detach().cpu().squeeze().numpy(), 80.0)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample, pred_np, pred_np_raw, target_np, out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
