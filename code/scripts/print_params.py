import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks._concentration import ConcentrationNet
from networks.fully_conv import FullyConv
from networks.unet_dav2 import UNetDav2


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print parameter counts for selected networks.")
    parser.add_argument("--input-channels", type=int, default=5)
    parser.add_argument("--unet-base-channels", type=int, default=16)
    parser.add_argument("--concentration-base-channels", type=int, default=16)
    parser.add_argument("--attention-method", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--dav2-encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = UNetDav2(
        input_channels=args.input_channels,
        unet_base_channels=args.unet_base_channels,
        dav2_encoder=args.dav2_encoder,
        freeze_dav2=True,
        device=torch.device("cpu"),
    )

    total, trainable = count_parameters(model.unet)
    print(f"UNet parameters: {total:,}")
    print(f"UNet trainable parameters: {trainable:,}")

    concentration = ConcentrationNet(
        in_channels=args.input_channels,
        base_channels=args.concentration_base_channels,
        attention_method=args.attention_method,
    )
    total, trainable = count_parameters(concentration)
    print(f"ConcentrationNet parameters: {total:,}")
    print(f"ConcentrationNet trainable parameters: {trainable:,}")

    fully_conv = FullyConv(in_channels=args.input_channels)
    total, trainable = count_parameters(fully_conv)
    print(f"FullyConv parameters: {total:,}")
    print(f"FullyConv trainable parameters: {trainable:,}")


if __name__ == "__main__":
    main()
