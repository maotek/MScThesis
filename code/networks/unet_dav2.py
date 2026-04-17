from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from networks.dav2 import Dav2



class _ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class _ConvBlock2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(torch.nn.Module):
    """Small UNet mapping event tensors to 3-channel DAV2 input."""

    def __init__(self, in_channels: int = 5, base_channels: int = 32, out_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base_channels)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = _ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = _ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_conv = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))
    

class SmallUNet2(torch.nn.Module):
    """Small UNet mapping event tensors to 3-channel DAV2 input."""

    def __init__(self, in_channels: int = 5, base_channels: int = 32, out_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = _ConvBlock2(in_channels, base_channels)
        self.enc2 = _ConvBlock2(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock2(base_channels * 2, base_channels * 4)
        self.dec2 = _ConvBlock2(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = _ConvBlock2(base_channels * 2 + base_channels, base_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_conv = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


class SmallUNet3(torch.nn.Module):
    """Small UNet variant with one encoder and one decoder stage."""

    def __init__(self, in_channels: int = 5, base_channels: int = 32, out_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base_channels)
        self.bottleneck = _ConvBlock(base_channels, base_channels * 2)
        self.dec1 = _ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_conv = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        b = self.bottleneck(self.pool(e1))

        d1 = F.interpolate(b, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


class UNetDav2(torch.nn.Module):
    """Pipeline: events -> small UNet -> DAV2 depth.

    The small UNet is trainable. DAV2 can be frozen/unfrozen with `freeze_dav2`.
    """

    def __init__(
        self,
        input_channels: int = 5,
        unet_base_channels: int = 32,
        unet_type: str = "small",
        dav2_encoder: str = "vits",
        dav2_checkpoint: Optional[str] = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        freeze_dav2: bool = True,
        device: torch.device = None,
        normalize_imagenet: bool = False,
        unet_output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.device = device
        self.freeze_dav2 = bool(freeze_dav2)
        self.in_channels = input_channels
        self.unet_output_channels = unet_output_channels
        self.unet_type = unet_type

        self.vis_temp = None

        if unet_type == "small":
            self.unet = SmallUNet(
                in_channels=input_channels,
                base_channels=unet_base_channels,
                out_channels=unet_output_channels
            )
        elif unet_type == "small2":
            self.unet = SmallUNet2(
                in_channels=input_channels,
                base_channels=unet_base_channels,
                out_channels=unet_output_channels
            )
        elif unet_type == "small3":
            self.unet = SmallUNet3(
                in_channels=input_channels,
                base_channels=unet_base_channels,
                out_channels=unet_output_channels
            )
        else:
            raise ValueError(f"Unsupported unet_type '{unet_type}'")

        if dav2_checkpoint is None:
            dav2_checkpoint = str(
                Path(__file__).resolve().parents[1]
                / "models"
                / "dav2"
                / "checkpoints"
                / f"depth_anything_v2_{dav2_encoder}.pth"
            )
        checkpoint_path = Path(dav2_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DAv2 checkpoint not found: {checkpoint_path}")

        self.dav2 = Dav2(
            encoder=dav2_encoder,
            checkpoint=str(checkpoint_path),
            device=self.device,
            input_size_width=input_size_width,
            input_size_height=input_size_height,
            normalize_imagenet=normalize_imagenet
        )

        self.set_dav2_frozen(self.freeze_dav2)

        self.to(self.device)

        print("[UNetDav2] DAv2 checkpoint:", str(checkpoint_path))
        print("[UNetDav2] DAv2 encoder:", dav2_encoder)
        print("[UNetDav2] DAv2 frozen:", self.freeze_dav2)
        print("[UNetDav2] Device:", self.device)
        print("[UNetDav2] UNet type:", unet_type)
        print("[UNetDav2] UNet output channels:", self.unet_output_channels)
        print("[UNetDav2] Normalize ImageNet:", normalize_imagenet)
        print("[UNetDav2] Input size (H,W):", (input_size_height, input_size_width))


    def set_dav2_frozen(self, freeze: bool = True) -> None:
        self.freeze_dav2 = bool(freeze)
        for param in self.dav2.parameters():
            param.requires_grad = not self.freeze_dav2

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        assert events.dim() == 4 and events.shape[1] == self.in_channels, f"Expected events of shape (B,{self.in_channels},H,W)"

        events = events.to(self.device)
        unet_rgb = self.unet(events)

        # If the UNet output is 1-channel, repeat it to make it 3-channel for DAV2
        if self.unet_output_channels == 1:
            unet_rgb = unet_rgb.repeat(1, 3, 1, 1)

        if True:
            self.vis_temp = unet_rgb.detach().cpu()

        depth = self.dav2(unet_rgb)
            
        return depth


class NewUNetDav2(torch.nn.Module):
    """Pipeline: events -> SmallUNet -> DAV2 inverse-depth -> metric depth.

    Unlike `UNetDav2`, this model performs the inverse-depth conversion internally
    using a learned scalar offset initialized to 1.0.
    """

    def __init__(
        self,
        input_channels: int = 5,
        unet_base_channels: int = 32,
        dav2_encoder: str = "vits",
        dav2_checkpoint: Optional[str] = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        freeze_dav2: bool = True,
        device: torch.device = None,
        normalize_imagenet: bool = False,
        unet_output_channels: int = 3,
        inv_depth_constant_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.freeze_dav2 = bool(freeze_dav2)
        self.in_channels = input_channels
        self.unet_output_channels = unet_output_channels

        self.vis_temp = None
        self.unet = SmallUNet(
            in_channels=input_channels,
            base_channels=unet_base_channels,
            out_channels=unet_output_channels,
        )
        self.inv_depth_constant = torch.nn.Parameter(torch.tensor(float(inv_depth_constant_init)))

        if dav2_checkpoint is None:
            dav2_checkpoint = str(
                Path(__file__).resolve().parents[1]
                / "models"
                / "dav2"
                / "checkpoints"
                / f"depth_anything_v2_{dav2_encoder}.pth"
            )
        checkpoint_path = Path(dav2_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DAv2 checkpoint not found: {checkpoint_path}")

        self.dav2 = Dav2(
            encoder=dav2_encoder,
            checkpoint=str(checkpoint_path),
            device=self.device,
            input_size_width=input_size_width,
            input_size_height=input_size_height,
            normalize_imagenet=normalize_imagenet,
        )

        self.set_dav2_frozen(self.freeze_dav2)

        self.to(self.device)

        print("[NewUNetDav2] DAv2 checkpoint:", str(checkpoint_path))
        print("[NewUNetDav2] DAv2 encoder:", dav2_encoder)
        print("[NewUNetDav2] DAv2 frozen:", self.freeze_dav2)
        print("[NewUNetDav2] Device:", self.device)
        print("[NewUNetDav2] UNet output channels:", self.unet_output_channels)
        print("[NewUNetDav2] Normalize ImageNet:", normalize_imagenet)
        print("[NewUNetDav2] Input size (H,W):", (input_size_height, input_size_width))
        print("[NewUNetDav2] Inverse-depth offset init:", float(inv_depth_constant_init))

    def set_dav2_frozen(self, freeze: bool = True) -> None:
        self.freeze_dav2 = bool(freeze)
        for param in self.dav2.parameters():
            param.requires_grad = not self.freeze_dav2

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        assert events.dim() == 4 and events.shape[1] == self.in_channels, f"Expected events of shape (B,{self.in_channels},H,W)"

        events = events.to(self.device)
        unet_rgb = self.unet(events)

        if self.unet_output_channels == 1:
            unet_rgb = unet_rgb.repeat(1, 3, 1, 1)

        self.vis_temp = unet_rgb.detach().cpu()

        inv_depth = self.dav2(unet_rgb)
        inv_depth_constant = torch.clamp(self.inv_depth_constant, min=1e-6)
        depth = 1.0 / (inv_depth + inv_depth_constant)
        return depth
