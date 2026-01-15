import argparse
import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import tqdm

from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.sbt.dsec_sequence import DsecSequence
from datasets.events import Histogram, Tencode
from networks.dav2_wrapper import Dav2Wrapper, Dav2InferWrapper
from evaluation import (
    add_to_metrics,
    prepare_target_data,
    prepare_target_data_torch,
)
from losses import normalized_depth_scale_and_shift


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
    parser.add_argument(
        "--representation",
        type=str,
        choices=("tencode", "histogram"),
        default="tencode",
        help="Event representation to use (tencode or histogram).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("dav2", "dav2_infer"),
        default="dav2",
        help="Model type: dav2 (standard forward) or dav2_infer (Imagenet-normalized infer_image).",
    )
    parser.add_argument("--clip-distance", type=float, default=80.0, help="Max depth value for metrics.")
    parser.add_argument("--use-scaleshift", action="store_true", default=True, help="Apply scale+shift alignment.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional CSV file to save metrics; defaults to output/evaluate_<model>_<rep>.csv",
    )
    parser.add_argument("--erase-csv", action="store_true", default=False, help="Remove existing CSV before writing.")
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


def make_representation(representation: str):
    representation = representation.lower()
    if representation == "tencode":
        return Tencode(height=DSEC_HEIGHT, width=DSEC_WIDTH, normalize=True, white_frame=True)
    if representation == "histogram":
        return Histogram(height=DSEC_HEIGHT, width=DSEC_WIDTH, remove_int_artifact=False)
    raise ValueError(f"Unsupported representation: {representation}")


def make_dataset(sequence_path: str, time_window_ms: int, representation: str) -> DsecSequence:

    rep = make_representation(representation)

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


def _write_csv_header(path: str, keys: Tuple[str, ...]) -> None:
    header = ["SEQ", "FRAMES", *[k.upper() for k in keys]]
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")


def _write_csv_row(path: str, seq: str, frames: int, keys: Tuple[str, ...], metrics: Dict[str, float]) -> None:
    row = [seq, str(frames), *[f"{metrics[k]:.6f}" for k in keys]]
    with open(path, "a") as f:
        f.write(",".join(row) + "\n")


def evaluate_sequence(
    seq_name: str,
    dsec_root: str,
    model: object,
    device: torch.device,
    time_window_ms: int,
    clip_distance: float,
    use_scaleshift: bool,
    representation: str,
) -> Tuple[Dict[str, float], int]:
    sequence_path = os.path.join(dsec_root, seq_name)
    if not os.path.isdir(sequence_path):
        raise FileNotFoundError(f"Sequence folder not found: {sequence_path}")
    dataset = make_dataset(sequence_path, time_window_ms, representation)

    metrics_sum: Dict[str, float] = {}
    num_frames = len(dataset)
    # num_frames = 50

    for idx in tqdm.tqdm(range(num_frames), desc=f"{seq_name}", leave=False):
        sample = dataset[idx]

        target_depth_t = sample["depth"][0].to(device)  # (1,H,W)
        events = sample["depth_aligned_events"][0].unsqueeze(0).to(device)
        
        if representation.lower() == "histogram" and events.shape[1] == 2:
            events = torch.cat([events, torch.zeros_like(events[:, :1])], dim=1)  # Pad histogram to 3 channels

        pred_depth = model(events)  # (1,1,H,W)
        pred_depth = pred_depth.squeeze(1)  # (1,H,W)

        target_proc_t = prepare_target_data_torch(target_depth_t, clip_distance)

        if use_scaleshift:
            scale, shift = normalized_depth_scale_and_shift(
                pred_depth, target_proc_t, target_proc_t > 0
            )
            pred_depth = scale[:, None, None] * pred_depth + shift[:, None, None]

        pred_np = np.clip(pred_depth.detach().cpu().squeeze().numpy(), 0, clip_distance)
        target_np = prepare_target_data(target_proc_t.detach().cpu().squeeze().numpy(), clip_distance)

        mask = np.ones_like(target_np, dtype=bool)

        metrics_sum = add_to_metrics(
            idx,
            metrics_sum,
            target_np,
            pred_np,
            mask,
            event_frame=None,
            prefix="_",
            debug=False,
            output_folder=None,
        )

        for depth_threshold in (10, 20, 30):
            threshold_mask = np.nan_to_num(target_np) < depth_threshold
            combined_mask = mask & threshold_mask
            metrics_sum = add_to_metrics(
                -1,
                metrics_sum,
                target_np,
                pred_np,
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

    # Default CSV path incorporates model and representation for clarity
    if args.csv_path is None:
        args.csv_path = os.path.join("output", f"evaluate_{args.model}_{args.representation}.csv")

    print(f"Evaluation using representation: {args.representation}, model: {args.model}, output CSV: {args.csv_path}, device: {device}")

    if args.csv_path and args.erase_csv and os.path.exists(args.csv_path):
        os.remove(args.csv_path)

    if args.model == "dav2":
        model = Dav2Wrapper(
            encoder="vitb",
            checkpoint=os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vitb.pth"),
            device=device,
            input_size=518,
        )
    elif args.model == "dav2_infer":
        model = Dav2InferWrapper(
            encoder="vitb",
            checkpoint=os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vitb.pth"),
            device=device,
            input_size=518,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

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
            use_scaleshift=args.use_scaleshift,
            representation=args.representation,
        )

        overall_sum = accumulate_metrics(overall_sum, metrics_sum)
        overall_frames += frames

        seq_avg = {k: v / frames for k, v in metrics_sum.items()}
        print(f"\nSequence {seq_name} ({frames} frames):")
        for k in sorted(seq_avg.keys()):
            print(f"  {k}: {seq_avg[k]:.6f}")

        if args.csv_path:
            keys = tuple(sorted(seq_avg.keys()))
            if not os.path.exists(args.csv_path):
                _write_csv_header(args.csv_path, keys)
            _write_csv_row(args.csv_path, seq_name, frames, keys, seq_avg)

    if overall_frames > 0:
        overall_avg = {k: v / overall_frames for k, v in overall_sum.items()}
        print("\n================ Overall (validation) ===============")
        print(f"Frames: {overall_frames}")
        for k in sorted(overall_avg.keys()):
            print(f"{k}: {overall_avg[k]:.6f}")

        if args.csv_path:
            keys = tuple(sorted(overall_avg.keys()))
            if not os.path.exists(args.csv_path):
                _write_csv_header(args.csv_path, keys)
            _write_csv_row(args.csv_path, "MEAN", overall_frames, keys, overall_avg)


if __name__ == "__main__":
    main()