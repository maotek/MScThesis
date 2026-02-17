import argparse
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from pathlib import Path

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import Tencode
from datasets.utils.data_augmentation import CenterCrop
from util import grayscale_to_uint8, rgb_to_uint8, save_image, save_rgb


def main():
    parser = argparse.ArgumentParser(description="Visualize DSEC sequence with Tencode events")
    parser.add_argument(
        "sequence_path",
        nargs="?",
        default="datasets/DSEC/data/validation/interlaken_00_f",
        help="Path to a DSEC sequence folder",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index in the sequence")
    parser.add_argument("--time-window-ms", type=int, default=50, help="Event window in milliseconds")
    parser.add_argument("--output-dir", default="output/test_dsec_sequence", help="Where to save PNGs")
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
        self_supervised=False,
    )

    sample = dataset[args.index]

    # Take first frame in the returned sequence
    events = sample["depth_aligned_events"][0]  # C,H,W
    depth = sample["depth"][0]  # 1,H,W or H,W
    print(f"Sample {args.index}: events shape {events.shape}, depth shape {depth.shape}")
    print(f"Event min/max: {events.min()}/{events.max()}, Depth min/max: {depth.min()}/{depth.max()}")

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

    os.makedirs(args.output_dir, exist_ok=True)
    # save raw and rectified tencode previews
    events_raw_rgb = rgb_to_uint8(tencode_raw)
    events_rect_rgb = rgb_to_uint8(tencode_rect)
    save_image(os.path.join(args.output_dir, f"events_tencode_raw_{args.index:05d}.png"), events_raw_rgb)
    save_image(os.path.join(args.output_dir, f"events_tencode_rect_{args.index:05d}.png"), events_rect_rgb)
    

    # Convert events (Tencode) to RGB for viewing
    events_rgb = rgb_to_uint8(events)

    # Depth visualization (simple min-max normalize)
    depth_np = depth.cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    depth_vis = depth_np.copy()
    if depth_vis.max() > depth_vis.min():
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
    else:
        depth_vis = np.zeros_like(depth_vis)
    depth_vis = grayscale_to_uint8(depth_vis)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # If RGB available, prepare it for visualization and save combined image
    rgb_vis = None
    if "rgb" in sample:
        rgb = sample["rgb"][0].cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_vis = rgb_to_uint8(rgb)

    # Save combined view (events | depth | rgb) when available
    combined_path = os.path.join(out_dir, f"combined_{args.index:05d}.png")
    if rgb_vis is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(events_rgb)
        axes[0].set_title("Events (Tencode)")
        axes[0].axis("off")
        axes[1].imshow(depth_vis, cmap="gray")
        axes[1].set_title("Depth")
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

    # Overlay events on depth visualization
    if depth_vis.shape[:2] != events_rgb.shape[:2]:
        events_for_overlay = cv2.resize(events_rgb, (depth_vis.shape[1], depth_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        events_for_overlay = events_rgb

    if depth_vis.ndim == 2:
        depth_for_overlay = np.repeat(depth_vis[..., None], 3, axis=2)
    elif depth_vis.ndim == 3 and depth_vis.shape[2] == 1:
        depth_for_overlay = np.repeat(depth_vis, 3, axis=2)
    else:
        depth_for_overlay = depth_vis

    depth_overlay = 0.2 * events_for_overlay.astype(np.float32) + 0.8 * depth_for_overlay.astype(np.float32)
    depth_overlay = np.clip(depth_overlay / 255.0, 0.0, 1.0)
    depth_overlay_path = os.path.join(out_dir, f"overlay_events_depth_{args.index:05d}.png")
    save_image(depth_overlay_path, depth_overlay)

    save_image(events_path, events_rgb)
    save_image(depth_path, depth_vis)
    print(f"Saved overlay (events on depth) to {depth_overlay_path}")

    if args.load_images == "yes" and "rgb" in sample:
        save_rgb(os.path.join(out_dir, f"rgb_{args.index:05d}.png"), sample["rgb"][0])

    print(f"Saved events to {events_path}")
    print(f"Saved depth to {depth_path}")


if __name__ == "__main__":
	main()
