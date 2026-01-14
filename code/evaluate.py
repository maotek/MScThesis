import argparse
import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events import Tencode
from networks.dav2_wrapper import Dav2Wrapper
from evaluation import add_to_metrics


def sanitize_prediction(prediction: np.ndarray, clip_distance: float) -> np.ndarray:
    """DAV2 outputs metric depth; just clamp and clean NaNs/Infs."""
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=clip_distance, neginf=0.0)
    return np.clip(prediction, 0, clip_distance)


def sanitize_target(target: np.ndarray, clip_distance: float) -> np.ndarray:
    """Keep GT in meters; clip and clean."""
    target = np.nan_to_num(target, nan=0.0, posinf=clip_distance, neginf=0.0)
    return np.clip(target, 0, clip_distance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DAV2 on DSEC validation sequences.")
    parser.add_argument(
        "--dsec-root",
        type=str,
        default="datasets/DSEC/data/validate",
        help="Root folder containing DSEC validation sequences (default: datasets/DSEC/data/validate)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="*",
        default=None,
        help="Specific sequence names to evaluate; if omitted, auto-detects subfolders in dsec-root.",
    )
    parser.add_argument("--time-window-ms", type=int, default=50, help="Event window size for tencode.")
    parser.add_argument("--clip-distance", type=float, default=80.0, help="Max depth value for metrics.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (cuda, mps, cpu). Auto-selects if not provided.",
    )
    return parser.parse_args()


def select_device(device_str: Optional[str]) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(sequence_path: str, time_window_ms: int) -> DsecSequence:

    rep = Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=True)

    dataset = DsecSequence(
        sequence_path=sequence_path,
        event_representation=rep,
        time_window_ms=time_window_ms,
        augmentator=None,
        load_images=False,
        overfit=False,
        sequence_window=1,
        sequence_step=1,
        split="validation",
        self_supervised=False,
        postfix="",
    )
    return dataset


def list_sequences(dsec_root: str) -> Tuple[str, ...]:
    if not os.path.isdir(dsec_root):
        raise FileNotFoundError(f"DSEC root not found: {dsec_root}")
    seqs = [name for name in os.listdir(dsec_root) if os.path.isdir(os.path.join(dsec_root, name))]
    seqs.sort()
    if not seqs:
        raise FileNotFoundError(f"No sequences found under {dsec_root}")
    return tuple(seqs)


def evaluate_sequence(
    seq_name: str,
    dsec_root: str,
    model: object,
    device: torch.device,
    time_window_ms: int,
    clip_distance: float,
) -> Tuple[Dict[str, float], int]:
    sequence_path = os.path.join(dsec_root, seq_name)
    if not os.path.isdir(sequence_path):
        raise FileNotFoundError(f"Sequence folder not found: {sequence_path}")
    dataset = make_dataset(sequence_path, time_window_ms)

    metrics_sum: Dict[str, float] = {}
    num_frames = len(dataset)

    preview_root = os.path.join("output", "eval_preview")

    def _save_depth(img: np.ndarray, path: str) -> None:
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imsave(path, img_norm, cmap="viridis")

    for idx in tqdm.tqdm(range(num_frames), desc=f"{seq_name}", leave=False):
        sample = dataset[idx]

        target_depth = sample["depth"][0].numpy()
        events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)

        pred_depth = model(events)
        pred_depth_np = pred_depth.squeeze(0).squeeze(0).detach().cpu().numpy()

        if idx == 0:
            before_dir = os.path.join(preview_root, "before_prep")
            os.makedirs(before_dir, exist_ok=True)
            _save_depth(pred_depth_np, os.path.join(before_dir, f"{seq_name}_pred.png"))
            _save_depth(target_depth.squeeze(), os.path.join(before_dir, f"{seq_name}_target.png"))

        print("min max before prep:", np.min(target_depth), np.max(target_depth), np.min(pred_depth_np), np.max(pred_depth_np))
        
        target_proc = sanitize_target(target_depth, clip_distance)
        pred_proc = sanitize_prediction(pred_depth_np, clip_distance)
        print("min max after prep:", np.min(target_proc), np.max(target_proc), np.min(pred_proc), np.max(pred_proc))

        if idx == 0:
            after_dir = os.path.join(preview_root, "after_prep")
            os.makedirs(after_dir, exist_ok=True)
            _save_depth(pred_proc.squeeze(), os.path.join(after_dir, f"{seq_name}_pred.png"))
            _save_depth(target_proc.squeeze(), os.path.join(after_dir, f"{seq_name}_target.png"))

        target_proc = np.squeeze(target_proc)
        pred_proc = np.squeeze(pred_proc)
        assert target_proc.ndim == 2 and pred_proc.ndim == 2, "Depth tensors must be 2D (H,W) after squeezing"
        mask = np.ones_like(target_proc, dtype=bool)

        metrics_sum = add_to_metrics(
            idx,
            metrics_sum,
            target_proc,
            pred_proc,
            mask,
            event_frame=None,
            prefix="_",
            debug=False,
            output_folder=None,
        )

        for depth_threshold in (10, 20, 30):
            threshold_mask = np.nan_to_num(target_proc) < depth_threshold
            combined_mask = mask & threshold_mask
            metrics_sum = add_to_metrics(
                -1,
                metrics_sum,
                target_proc,
                pred_proc,
                combined_mask,
                event_frame=None,
                prefix=f"_{depth_threshold}_",
                debug=False,
                output_folder=None,
            )

    return metrics_sum, num_frames


def accumulate_metrics(target: Dict[str, float], source: Dict[str, float]) -> Dict[str, float]:
    for k, v in source.items():
        target[k] = target.get(k, 0.0) + v
    return target


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    model = Dav2Wrapper(
        encoder="vitb",
        checkpoint=os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vitb.pth"),
        device=device,
        input_size=518,
    )

    overall_sum: Dict[str, float] = {}
    overall_frames = 0

    seq_list = args.sequences if args.sequences is not None else list_sequences(args.dsec_root)

    for seq_name in seq_list:
        metrics_sum, frames = evaluate_sequence(
            seq_name=seq_name,
            dsec_root=args.dsec_root,
            model=model,
            device=device,
            time_window_ms=args.time_window_ms,
            clip_distance=args.clip_distance,
        )

        overall_sum = accumulate_metrics(overall_sum, metrics_sum)
        overall_frames += frames

        seq_avg = {k: v / frames for k, v in metrics_sum.items()}
        print(f"\nSequence {seq_name} ({frames} frames):")
        for k in sorted(seq_avg.keys()):
            print(f"  {k}: {seq_avg[k]:.6f}")

    if overall_frames > 0:
        overall_avg = {k: v / overall_frames for k, v in overall_sum.items()}
        print("\n================ Overall (validation) ===============")
        print(f"Frames: {overall_frames}")
        for k in sorted(overall_avg.keys()):
            print(f"{k}: {overall_avg[k]:.6f}")


if __name__ == "__main__":
    main()