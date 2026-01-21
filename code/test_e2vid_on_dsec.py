import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import VoxelGrid
from networks.e2vid_wrapper import load_e2vid
from util import save_grayscale, save_voxelgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2VID on a DSEC sequence and visualize intensity reconstruction.")
    parser.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default="datasets/DSEC/data/train/zurich_city_01_e",
        help="Path to a DSEC sequence root (default: datasets/DSEC/data/train/zurich_city_01_e)",
    )
    parser.add_argument("--index", type=int, default=10, help="Index within the sequence to visualize (depth-aligned events).")
    parser.add_argument("--time-window-ms", type=int, default=50, help="Event window size for building voxel-grid representations.")
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins for the voxel grid; must match the checkpoint's num_bins (E2VID lightweight uses 5).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional E2VID checkpoint path. Defaults to models/rpg_e2vid/pretrained/E2VID_lightweight.pth.tar",
    )
    parser.add_argument("--output-dir", type=str, default="output/test_e2vid_on_dsec", help="Where to save visualizations.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/mps/cpu). Defaults to auto-selection.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def visualize(events_voxel: torch.Tensor, recon: torch.Tensor, out_dir: str, idx: int) -> None:
    # events_voxel: (C,H,W), recon: (1,H,W)
    save_voxelgrid(os.path.join(out_dir, f"{idx:05d}_events.png"), events_voxel)
    save_grayscale(os.path.join(out_dir, f"{idx:05d}_recon.png"), recon)


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
        load_images=False,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        split="train",
        self_supervised=False,
        postfix="",
    )

    assert 0 <= args.index < len(dataset), f"Index {args.index} out of range for sequence of length {len(dataset)}"
    sample = dataset[args.index]

    model = load_e2vid(weights_path=args.checkpoint, device=device)
    # E2VID expects (B,C,H,W)
    events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
    recon = model(events)  # (B,1,H,W)

    out_dir = ensure_dir(args.output_dir)
    visualize(sample["depth_aligned_events"][0], recon[0], out_dir, args.index)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
