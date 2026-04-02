import argparse
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from datasets.DSEC.constants import DSEC_HEIGHT, DSEC_WIDTH
from datasets.DSEC.dsec_dataset import fetch_dataloader as fetch_dsec_dataloader
from networks.dae import DAE
from networks.fully_conv_dav2 import FullyConvDav2
from networks.unet_dav2 import UNetDav2

BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(BASE_DIR / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure computational efficiency (FLOPs + timing) for different training setups "
            "on a single DSEC sequence."
        )
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="datasets/DSEC/data",
        help="Root path to DSEC dataset (contains train/validation/test folders).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Which DSEC split to evaluate (train/validation/test).",
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        default="",
        help="Optional sequence name to benchmark (defaults to first common sequence).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to average timings over.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Warmup frames (not included in timing).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of bins/channels for voxel grid (FullyConv/UNet input).",
    )
    parser.add_argument(
        "--dae-checkpoint",
        type=str,
        default=os.path.join(
            "models",
            "depthanyevent",
            "weights",
            "dav2",
            "finetuned_dsec",
            "finetuned_dsec.pth",
        ),
        help="DAE pretrained checkpoint (DSEC finetuned).",
    )
    parser.add_argument(
        "--dae-nopretrain",
        action="store_true",
        help="Skip loading DAE checkpoint (use random weights) if checkpoint not present.",
    )
    parser.add_argument(
        "--dav2-checkpoint",
        type=str,
        default=os.path.join("models", "dav2", "checkpoints", "depth_anything_v2_vits.pth"),
        help="Pretrained DAv2 checkpoint.",
    )
    parser.add_argument(
        "--fully-conv-checkpoint",
        type=str,
        default="train_output/train_dsec_fully_conv_dav2_batch10_RC/epoch_050.pt",
        help="Checkpoint containing trained FullyConv weights (optional).",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=str,
        default="train_output/train_dsec_unet_dav2_batch10_ch16/epoch_050.pt",
        help="Checkpoint containing trained UNet weights (optional).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Optional device override (cuda, mps, cpu).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output/test_computational_efficiency/summary.md",
        help="Where to save the summary table (markdown).",
    )
    return parser.parse_args()


def pick_device(device_str: str) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_dataloaders(datapath: str, split: str, num_bins: int) -> Tuple[dict, dict]:
    preprocess_config = [
        {"preprocessing_type": "CenterCrop", "height": 320, "width": 640}
    ]

    data_loader_config_dae = {
        "dataset": "dsec",
        "datapath": datapath,
        "split": split,
        "concatenate_sequences": False,
        "event_representation": {
            "representation_type": "tencode",
            "normalize": True,
            "white_frame": False,
            "height": DSEC_HEIGHT,
            "width": DSEC_WIDTH,
        },
        "preprocessing": preprocess_config,
        "load_images": False,
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": False,
        "sequence_window": 1,
        "sequence_step": 1,
        "time_window_ms": 50,
    }

    data_loader_config_voxel = {
        "dataset": "dsec",
        "datapath": datapath,
        "split": split,
        "concatenate_sequences": False,
        "event_representation": {
            "representation_type": "voxelgrid",
            "channels": num_bins,
            "normalize": True,
            "height": DSEC_HEIGHT,
            "width": DSEC_WIDTH,
        },
        "preprocessing": preprocess_config,
        "load_images": False,
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": False,
        "sequence_window": 1,
        "sequence_step": 1,
        "time_window_ms": 50,
    }

    data_loaders_dae = fetch_dsec_dataloader(data_loader_config_dae, test=True)
    data_loaders_voxel = fetch_dsec_dataloader(data_loader_config_voxel, test=True)
    return data_loaders_dae, data_loaders_voxel


def pick_sequence_name(seq_names: List[str], requested: str) -> str:
    if requested:
        if requested not in seq_names:
            raise ValueError(f"Requested sequence '{requested}' not found in dataloaders.")
        return requested
    return seq_names[0]


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def maybe_load_fully_conv_weights(model: FullyConvDav2, ckpt_path: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        print(f"[WARN] FullyConv checkpoint not found: {ckpt_path} (using random weights)")
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    fc_state = {k.replace("fully_conv.", ""): v for k, v in state.items() if k.startswith("fully_conv.")}
    if not fc_state:
        print(f"[WARN] No FullyConv weights in checkpoint: {ckpt_path} (using random weights)")
        return
    model.fully_conv.load_state_dict(fc_state, strict=True)


def maybe_load_unet_weights(model: UNetDav2, ckpt_path: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        print(f"[WARN] UNet checkpoint not found: {ckpt_path} (using random weights)")
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    unet_state = {k.replace("unet.", ""): v for k, v in state.items() if k.startswith("unet.")}
    if not unet_state:
        print(f"[WARN] No UNet weights in checkpoint: {ckpt_path} (using random weights)")
        return
    model.unet.load_state_dict(unet_state, strict=True)


def estimate_flops(
    forward_fn: Callable[[], torch.Tensor], device: torch.device
) -> Tuple[Optional[float], Optional[str]]:
    try:
        import torch.profiler as profiler

        activities = [profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(profiler.ProfilerActivity.CUDA)
        with profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            forward_fn()
        flops = 0
        for evt in prof.key_averages():
            if evt.flops is not None:
                flops += evt.flops
        if flops == 0:
            return None, "Profiler returned 0 FLOPs"
        return float(flops), None
    except Exception as exc:
        return None, str(exc)


def run_timing(
    name: str,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    input_getter: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    max_frames: int,
    warmup_frames: int,
) -> Dict[str, object]:
    model = model.to(device)
    model.eval()

    # Grab one sample for FLOPs
    first_sample = next(iter(loader))
    events = input_getter(first_sample).to(device)

    def forward_once():
        with torch.no_grad():
            _ = model(events)
        return _

    flops, flops_err = estimate_flops(forward_once, device)

    # Warmup
    warmup = min(warmup_frames, len(loader))
    if warmup > 0:
        for idx, sample in enumerate(loader):
            if idx >= warmup:
                break
            _ = model(input_getter(sample).to(device))
        sync_device(device)

    # Forward-only timing
    total_time = 0.0
    count = 0
    for idx, sample in enumerate(loader):
        if idx < warmup:
            continue
        if count >= max_frames:
            break
        x = input_getter(sample).to(device)
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        sync_device(device)
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        count += 1

    forward_ms = (total_time / max(count, 1)) * 1000.0

    # Train-step timing (forward + backward)
    model.train()
    total_time = 0.0
    count = 0
    for idx, sample in enumerate(loader):
        if idx < warmup:
            continue
        if count >= max_frames:
            break
        x = input_getter(sample).to(device)
        sync_device(device)
        t0 = time.perf_counter()
        out = model(x)
        loss = out.mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        sync_device(device)
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        count += 1

    train_step_ms = (total_time / max(count, 1)) * 1000.0

    total_params, trainable_params = count_params(model)

    return {
        "name": name,
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "flops_err": flops_err,
    }


def main() -> None:
    args = parse_args()

    device = pick_device(args.device)

    args.datapath = resolve_path(args.datapath)
    args.dae_checkpoint = resolve_path(args.dae_checkpoint)
    args.dav2_checkpoint = resolve_path(args.dav2_checkpoint)
    args.fully_conv_checkpoint = resolve_path(args.fully_conv_checkpoint)
    args.unet_checkpoint = resolve_path(args.unet_checkpoint)
    args.output_path = resolve_path(args.output_path)

    # Build dataloaders (matching fully_conv_test settings)
    data_loaders_dae, data_loaders_voxel = build_dataloaders(
        args.datapath, args.split, args.num_bins
    )

    seq_names = sorted(set(data_loaders_dae.keys()) & set(data_loaders_voxel.keys()))
    if not seq_names:
        raise FileNotFoundError("No sequences found in dataloaders.")

    seq_name = pick_sequence_name(seq_names, args.sequence_name)
    loader_dae = data_loaders_dae[seq_name]
    loader_voxel = data_loaders_voxel[seq_name]

    print(f"Using device: {device}")
    print(f"Sequence: {seq_name}")
    print(f"DAE loader frames: {len(loader_dae)} | Voxel loader frames: {len(loader_voxel)}")

    # --- Build models ---
    # DAE full finetune (unfrozen encoder)
    dae_full = DAE(
        encoder="vits",
        checkpoint=args.dae_checkpoint,
        device=device,
        input_size_width=350,
        input_size_height=266,
        activation="relu",
        scale_factor=1.0,
        inv_prediction=True,
        freeze_encoder=False,
        input_channels=3,
        nopretrain=args.dae_nopretrain,
    )

    # FullyConv + DAv2
    fc_full = FullyConvDav2(
        input_channels=args.num_bins,
        dav2_encoder="vits",
        dav2_checkpoint=args.dav2_checkpoint,
        input_size_width=350,
        input_size_height=266,
        freeze_dav2=False,
        device=device,
    )
    maybe_load_fully_conv_weights(fc_full, args.fully_conv_checkpoint)

    fc_frozen = FullyConvDav2(
        input_channels=args.num_bins,
        dav2_encoder="vits",
        dav2_checkpoint=args.dav2_checkpoint,
        input_size_width=350,
        input_size_height=266,
        freeze_dav2=True,
        device=device,
    )
    maybe_load_fully_conv_weights(fc_frozen, args.fully_conv_checkpoint)

    # UNet + DAv2
    unet_full = UNetDav2(
        input_channels=args.num_bins,
        unet_base_channels=16,
        unet_type="small",
        dav2_encoder="vits",
        dav2_checkpoint=args.dav2_checkpoint,
        input_size_width=350,
        input_size_height=266,
        freeze_dav2=False,
        device=device,
    )
    maybe_load_unet_weights(unet_full, args.unet_checkpoint)

    unet_frozen = UNetDav2(
        input_channels=args.num_bins,
        unet_base_channels=16,
        unet_type="small",
        dav2_encoder="vits",
        dav2_checkpoint=args.dav2_checkpoint,
        input_size_width=350,
        input_size_height=266,
        freeze_dav2=True,
        device=device,
    )
    maybe_load_unet_weights(unet_frozen, args.unet_checkpoint)

    # Input getters
    def get_events_dae(sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        # depth_aligned_events: [B, T, C, H, W] -> use B=1, T=1
        return sample["depth_aligned_events"][:, 0]

    def get_events_voxel(sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sample["depth_aligned_events"][:, 0]

    # Wrap DAE to use underlying model without @no_grad
    class _DAEForward(torch.nn.Module):
        def __init__(self, dae: DAE):
            super().__init__()
            self.dae = dae

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dae.model.infer_image(x)

    dae_forward = _DAEForward(dae_full)

    # Run timings
    results = []

    results.append(
        run_timing(
            name="DAE_full_finetune",
            model=dae_forward,
            loader=loader_dae,
            device=device,
            input_getter=get_events_dae,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
    )

    results.append(
        run_timing(
            name="FullyConv_full_finetune",
            model=fc_full,
            loader=loader_voxel,
            device=device,
            input_getter=get_events_voxel,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
    )

    results.append(
        run_timing(
            name="FullyConv_repr_only",
            model=fc_frozen,
            loader=loader_voxel,
            device=device,
            input_getter=get_events_voxel,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
    )

    results.append(
        run_timing(
            name="UNet_full_finetune",
            model=unet_full,
            loader=loader_voxel,
            device=device,
            input_getter=get_events_voxel,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
    )

    results.append(
        run_timing(
            name="UNet_repr_only",
            model=unet_frozen,
            loader=loader_voxel,
            device=device,
            input_getter=get_events_voxel,
            max_frames=args.max_frames,
            warmup_frames=args.warmup_frames,
        )
    )

    # Print summary table
    print("\nSummary (per-frame averages)")
    header = (
        "| Model | Forward ms | Train-step ms | Trainable params | Total params | FLOPs |"
    )
    sep = "|---|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)

    table_lines = [header, sep]
    for r in results:
        flops_str = "N/A"
        if r["flops"] is not None:
            flops_str = f"{r['flops'] / 1e9:.3f} GFLOPs"
        elif r["flops_err"]:
            flops_str = f"N/A ({r['flops_err']})"
        line = (
            "| {name} | {fwd:.3f} | {train:.3f} | {trainable:,} | {total:,} | {flops} |".format(
                name=r["name"],
                fwd=r["forward_ms"],
                train=r["train_step_ms"],
                trainable=r["trainable_params"],
                total=r["total_params"],
                flops=flops_str,
            )
        )
        print(line)
        table_lines.append(line)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
