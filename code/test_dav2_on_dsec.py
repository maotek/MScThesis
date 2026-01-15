import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events import Tencode
from networks.dav2_wrapper import Dav2Wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DAV2 on a DSEC sequence and visualize depth.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/train/interlaken_00_c",
        help="Path to a DSEC sequence root (default: datasets/DSEC/data/train/interlaken_00_c)",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Index within the sequence to visualize (depth-aligned events)."
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


def visualize(sample: dict, depth_pred: torch.Tensor, out_dir: str, idx: int) -> None:
    events = sample["depth_aligned_events"][0]
    events_rgb = to_rgb_events(events)

    depth_np = depth_pred.squeeze().detach().cpu().numpy()
    depth_rgb = depth_to_colormap(depth_np)

    if "depth_aligned_rgb" in sample and sample["depth_aligned_rgb"] is not None:
        rgb = sample["depth_aligned_rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
        rgb = (255 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
    else:
        rgb = None

    plt.imsave(os.path.join(out_dir, f"{idx:05d}_events.png"), events_rgb)
    plt.imsave(os.path.join(out_dir, f"{idx:05d}_depth.png"), depth_rgb)
    if rgb is not None:
        plt.imsave(os.path.join(out_dir, f"{idx:05d}_rgb.png"), rgb)


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

    model = Dav2Wrapper(encoder=args.encoder, checkpoint=args.checkpoint, device=device, input_size=args.input_size)

    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    depth_pred = model(events)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample, depth_pred, out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
