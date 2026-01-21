import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from networks.dav2_wrapper import Dav2Wrapper
from datasets.events import Tencode
from evaluation import prepare_target_data_torch, prepare_target_data
from losses import normalized_depth_scale_and_shift


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
        default="vitb",
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


def to_rgb_image(rgb_tensor: torch.Tensor) -> np.ndarray:
    """Convert RGB tensor (3,H,W) to RGB uint8 image."""
    rgb_np = rgb_tensor.detach().cpu().numpy()
    rgb_np = np.transpose(rgb_np, (1, 2, 0))  # HWC
    rgb_np = (255 * np.clip(rgb_np, 0.0, 1.0)).astype(np.uint8)
    return rgb_np


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    cmap = plt.get_cmap("viridis")
    depth_rgb = (255 * cmap(depth_norm)[..., :3]).astype(np.uint8)
    return depth_rgb


def visualize(sample: dict, pred_np: np.ndarray, pred_np_raw: np.ndarray, target_np: np.ndarray, out_dir: str, idx: int) -> None:
    rgb = sample["rgb"][0]
    input_rgb = to_rgb_image(rgb)

    pred_rgb = depth_to_colormap(pred_np)
    pred_raw_rgb = depth_to_colormap(pred_np_raw)
    gt_rgb = depth_to_colormap(target_np)

    # Overlay: blend scaled prediction and ground truth
    overlay_rgb = 0.5 * pred_rgb + 0.5 * gt_rgb

    plt.imsave(os.path.join(out_dir, f"{idx:05d}_input.png"), input_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_pred_raw_depth.png"), pred_raw_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_pred_scaled_depth.png"), pred_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_gt_depth.png"), gt_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_overlay.png"), overlay_rgb.astype(np.uint8))


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

    model = Dav2Wrapper(encoder=args.encoder, checkpoint=args.checkpoint, device=device, input_size=args.input_size)

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