import os
from pathlib import Path
import numpy as np
import cv2
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import Tencode
from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from util import rgb_to_uint8, save_image

def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def main():
    sequence_path = "datasets/DSEC/data/validate/interlaken_00_g"  # Change as needed
    output_dir = ensure_dir("output/test_align_rgb_to_event")
    idx = 100  # Change as needed

    # Load sequence
    rep = Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=False)
    dataset = DsecSequence(
        sequence_path=sequence_path,
        event_representation=rep,
        time_window_ms=50,
        load_images=True,
        augmentator=None,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        split="train",
        self_supervised=False,
        postfix="",
    )
    sample = dataset[idx]

    # RGB is already warped inside DsecSequence.
    rgb_tensor = sample["rgb"][0]  # C, H, W in [0, 1]
    rgb_img = (rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    # Load original (unwarped) RGB directly from the folder
    start_index = idx * dataset.sequence_step
    raw_path = os.path.join(dataset.base_left_images_path, dataset.left_images[start_index])
    raw_bgr = cv2.imread(raw_path)
    if raw_bgr is None:
        raise ValueError(f"Failed to load original RGB image: {raw_path}")
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # Load tencode image aligned to RGB timestamps (T, 3, H, W)
    tencode_tensor = sample["rgb_aligned_events"][0]
    tencode_img = rgb_to_uint8(tencode_tensor)

    # Save for visualization
    aligned_path = os.path.join(output_dir, f"{idx:05d}_aligned_rgb.png")
    raw_path_out = os.path.join(output_dir, f"{idx:05d}_rgb_raw.png")
    tencode_path = os.path.join(output_dir, f"{idx:05d}_tencode.png")
    overlay_path = os.path.join(output_dir, f"{idx:05d}_rgb_tencode_overlay.png")

    # Overlay in event camera frame (tencode matches event resolution)
    overlay = cv2.addWeighted(rgb_img, 0.6, tencode_img, 0.4, 0.0)

    save_image(aligned_path, rgb_img)
    save_image(raw_path_out, raw_rgb)
    save_image(tencode_path, tencode_img)
    save_image(overlay_path, overlay)
    print(f"Saved aligned RGB to {aligned_path}")
    print(f"Saved raw RGB to {raw_path_out}")
    print(f"Saved tencode to {tencode_path}")
    print(f"Saved overlay to {overlay_path}")

if __name__ == "__main__":
    main()
