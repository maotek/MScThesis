import numpy as np
import torch

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader

DATA_LOADER_CONFIG = {
    "dataset": "dsec",
    "datapath": "datasets/DSEC/data",
    "split": "validation",
    "concatenate_sequences": False,
    "event_representation": {
        "representation_type": "tencode",
        "normalize": True,
        "white_frame": False,
        "height": DSEC_HEIGHT,
        "width": DSEC_WIDTH,
    },
    "preprocessing": [
        {
            "preprocessing_type": "CenterCrop",
            "height": 320,
            "width": 640,
        }
    ],
    "load_images": True,
    "batch_size": 1,
    "num_workers": 0,
    "pin_memory": False,
    "shuffle": False,
    "sequence_window": 1,
    "sequence_step": 1,
    "time_window_ms": 50,
}

def main() -> None:
    dataloaders = fetch_dsec_dataloader(DATA_LOADER_CONFIG, test=True)
    if not dataloaders:
        raise RuntimeError("No DSEC dataloaders created. Check datapath/split.")

    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    total_frames = 0

    for seq_name, loader in dataloaders.items():
        seq_count = 0
        for sample in loader:
            if "rgb" not in sample:
                raise KeyError("Dataloader sample missing 'rgb'. Ensure load_images=True.")
            rgb = sample["rgb"][:, 0]  # (B,C,H,W), B=1
            rgb_np = rgb.squeeze(0).permute(1, 2, 0).numpy()  # H,W,C

            h, w = rgb_np.shape[:2]
            total_pixels += h * w
            total_frames += 1
            seq_count += 1

            sum_c += rgb_np.reshape(-1, 3).sum(axis=0, dtype=np.float64)
            sum_sq_c += (rgb_np.reshape(-1, 3) ** 2).sum(axis=0, dtype=np.float64)

        print(f"[{seq_name}] {seq_count} frames")

    if total_pixels == 0:
        raise RuntimeError("No pixels processed. Check dataset path and split.")

    mean = sum_c / total_pixels
    var = sum_sq_c / total_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    print("\n=== DSEC RGB Stats ===")
    print(f"Split: {DATA_LOADER_CONFIG['split']}")
    if DATA_LOADER_CONFIG.get("preprocessing"):
        for prep in DATA_LOADER_CONFIG["preprocessing"]:
            if prep.get("preprocessing_type") == "CenterCrop":
                print(f"Center crop: {prep.get('height')}x{prep.get('width')}")
                break
        else:
            print("Center crop: None")
    else:
        print("Center crop: None")
    print(f"Frames: {total_frames}")
    print(f"Pixels: {total_pixels}")
    print(f"Mean (R,G,B): {mean.tolist()}")
    print(f"Std  (R,G,B): {std.tolist()}")
    print(f"Var  (R,G,B): {var.tolist()}")


if __name__ == "__main__":
    main()
