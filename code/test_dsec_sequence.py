import argparse
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from pathlib import Path

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import Tencode, Histogram, VoxelGrid
from datasets.utils.data_augmentation import CenterCrop


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
	arr = np.clip(arr, 0.0, 1.0)
	arr = (arr * 255.0).astype(np.uint8)
	return arr


def save_image(path: str, img: np.ndarray):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	plt.imsave(path, img)


def main():
    parser = argparse.ArgumentParser(description="Visualize DSEC sequence with Tencode events")
    parser.add_argument(
        "sequence_path",
        nargs="?",
        default="datasets/DSEC/data/train/interlaken_00_c",
        help="Path to a DSEC sequence folder (default: datasets/DSEC/data/train/interlaken_00_c)",
    )
    parser.add_argument("--index", type=int, default=100, help="Sample index in the sequence")
    parser.add_argument("--time-window-ms", type=int, default=50, help="Event window in milliseconds")
    parser.add_argument("--output-dir", default="output/dsec_sequence_vis", help="Where to save PNGs")
    parser.add_argument(
        "--load-images",
        choices=["yes", "no"],
        default="yes",
        help="Load RGB frames: 'yes' loads rectified images, 'no' disables",
    )
    parser.add_argument(
        "--center-crop",
        choices=["yes", "no"],
        default="no",
        help="Apply CenterCrop using DSEC_HEIGHT x DSEC_WIDTH from Augmentator (yes/no)",
    )
    parser.add_argument("--white-frame", action="store_true", help="Use white background for empty Tencode")
    args = parser.parse_args()

    # Build Tencode representation and dataset
    rep = Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=args.white_frame)

    # Build optional augmentator (center-crop) if requested
    augmentator = None
    if args.center_crop == "yes":
        augmentator = CenterCrop(DSEC_HEIGHT, DSEC_WIDTH)

    dataset = DsecSequence(
        sequence_path=args.sequence_path,
        event_representation=rep,
        time_window_ms=args.time_window_ms,
        augmentator=augmentator,
        load_images=args.load_images == "yes",
        sequence_window=1,
        sequence_step=1,
        split="train",
        self_supervised=False,
    )

    sample = dataset[args.index]

    # Take first frame in the returned sequence
    events = sample["depth_aligned_events"][0]  # C,H,W
    depth = sample["depth"][0]  # 1,H,W or H,W
    print(f"Sample {args.index}: events shape {events.shape}, depth shape {depth.shape}")

    # --- Quick test: save Tencode with and without rectification ---
    # Fetch raw events for this timestamp window directly from the EventSlicer
    ev_window = dataset.disparity_aligned_event_windows[args.index]
    raw_events = dataset.event_slicer["left"].get_events(*ev_window)
    xr = raw_events["x"]
    yr = raw_events["y"]
    pr = raw_events["p"]
    tr = raw_events["t"]

    # Convert to Tencode (no rectification)
    tencode_raw = dataset.events_to_representation(xr, yr, pr, tr)

    # Convert to Tencode (rectified coordinates)
    xy_rect = dataset.rectify_events(xr, yr, "left")
    x_rect = xy_rect[:, 0]
    y_rect = xy_rect[:, 1]
    tencode_rect = dataset.events_to_representation(x_rect, y_rect, pr, tr)

    def _rep_to_uint8(img):
        # handle torch tensor, numpy array or other sequence
        if hasattr(img, "cpu"):
            arr = img.cpu().numpy()
        elif isinstance(img, np.ndarray):
            arr = img
        else:
            arr = np.asarray(img)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return to_uint8_rgb(arr)

    os.makedirs(args.output_dir, exist_ok=True)
    # save raw and rectified tencode previews
    events_raw_rgb = rep.to_rgb_mono(tencode_raw)
    events_rect_rgb = rep.to_rgb_mono(tencode_rect)
    events_raw_rgb = _rep_to_uint8(events_raw_rgb)
    events_rect_rgb = _rep_to_uint8(events_rect_rgb)
    save_image(os.path.join(args.output_dir, f"events_tencode_raw_{args.index:05d}.png"), events_raw_rgb)
    save_image(os.path.join(args.output_dir, f"events_tencode_rect_{args.index:05d}.png"), events_rect_rgb)
    

    # Convert events (Tencode) to RGB for viewing
    events_rgb = rep.to_rgb_mono(events)
    # handle torch tensor, numpy array or other sequence
    if hasattr(events_rgb, "cpu"):
        events_rgb = events_rgb.cpu().numpy()
    elif isinstance(events_rgb, np.ndarray):
        events_rgb = events_rgb
    else:
        events_rgb = np.asarray(events_rgb)
    if events_rgb.max() > 1.0:
        events_rgb = events_rgb / 255.0
    events_rgb = to_uint8_rgb(events_rgb)

    # Depth visualization (simple min-max normalize)
    depth_np = depth.cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    depth_vis = depth_np.copy()
    print(depth_vis.shape, depth_vis.dtype, np.min(depth_vis), np.max(depth_vis))
    if depth_vis.max() > depth_vis.min():
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
    else:
        depth_vis = np.zeros_like(depth_vis)
    depth_vis = to_uint8_rgb(depth_vis)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # If RGB available, prepare it for visualization and save combined image
    rgb_vis = None
    if "rgb" in sample:
        rgb = sample["rgb"][0].cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_vis = to_uint8_rgb(rgb.transpose(1, 2, 0))
        print(f"RGB frame shape: {rgb_vis.shape}")

    # Save combined view (events | depth | rgb) when available
    combined_path = os.path.join(out_dir, f"combined_{args.index:05d}.png")
    if rgb_vis is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(events_rgb)
        axes[0].set_title("Events (Tencode)")
        axes[0].axis("off")
        axes[1].imshow(depth_vis, cmap="gray")
        axes[1].set_title("Depth (min-max)")
        axes[1].axis("off")
        axes[2].imshow(rgb_vis)
        axes[2].set_title("RGB (nearest)")
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(combined_path, dpi=150)
        plt.close(fig)
        print(f"Saved combined view to {combined_path}")

        # Overlay events on RGB with alpha; resize events to RGB resolution if needed
        if rgb_vis.shape[:2] != events_rgb.shape[:2]:
            events_for_overlay = cv2.resize(events_rgb, (rgb_vis.shape[1], rgb_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            events_for_overlay = events_rgb
        overlay = 0.5 * events_for_overlay.astype(np.float32) + 0.5 * rgb_vis.astype(np.float32)
        overlay = np.clip(overlay / 255.0, 0.0, 1.0)
        overlay_path = os.path.join(out_dir, f"overlay_events_rgb_{args.index:05d}.png")
        save_image(overlay_path, overlay)
        print(f"Saved overlay to {overlay_path}")

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, f"events_{args.index:05d}.png")
    depth_path = os.path.join(out_dir, f"depth_{args.index:05d}.png")

    save_image(events_path, events_rgb)
    save_image(depth_path, depth_vis)

    if args.load_images == "yes" and "rgb" in sample:
        rgb = sample["rgb"][0].cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        save_image(os.path.join(out_dir, f"rgb_{args.index:05d}.png"), rgb.transpose(1, 2, 0))

    print(f"Saved events to {events_path}")
    print(f"Saved depth to {depth_path}")


if __name__ == "__main__":
	main()

