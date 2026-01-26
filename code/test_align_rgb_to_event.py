import os
from pathlib import Path
import numpy as np
import cv2
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events.events_representations import Tencode
from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from util import tencode_to_uint8

def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def main():
    sequence_path = "datasets/DSEC/data/validate/interlaken_00_c"  # Change as needed
    output_dir = ensure_dir("output/test_align_rgb_to_event")
    idx = 60  # Change as needed

    # Load sequence
    rep = Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True)
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

    # Load tencode image aligned to RGB timestamps (T, 3, H, W)
    tencode_tensor = sample["rgb_aligned_events"][0]
    tencode_img = tencode_to_uint8(tencode_tensor)

    # Save for visualization
    aligned_path = os.path.join(output_dir, f"{idx:05d}_aligned_rgb.png")
    rgb_path = os.path.join(output_dir, f"{idx:05d}_rgb.png")
    tencode_path = os.path.join(output_dir, f"{idx:05d}_tencode.png")
    overlay_path = os.path.join(output_dir, f"{idx:05d}_rgb_tencode_overlay.png")

    # Overlay in event camera frame (tencode matches event resolution)
    overlay = cv2.addWeighted(rgb_img, 0.6, tencode_img, 0.4, 0.0)

    cv2.imwrite(aligned_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(tencode_path, cv2.cvtColor(tencode_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved aligned RGB to {aligned_path}")
    print(f"Saved RGB to {rgb_path}")
    print(f"Saved tencode to {tencode_path}")
    print(f"Saved overlay to {overlay_path}")

if __name__ == "__main__":
    main()
