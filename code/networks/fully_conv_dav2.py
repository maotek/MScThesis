import torch
from pathlib import Path
from typing import Optional

from networks.dav2 import Dav2
from networks.fully_conv import FullyConv


class FullyConvDav2(torch.nn.Module):
    """
    Pipeline: events -> FullyConv -> DAV2 depth.
    The FullyConv network is trainable. DAV2 can be frozen/unfrozen with `freeze_dav2`.
    """

    def __init__(
        self,
        input_channels: int = 5,
        dav2_encoder: str = "vits",
        dav2_checkpoint: Optional[str] = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        freeze_dav2: bool = True,
        device: torch.device = None,
        fc_output_channels: int = 3,
        normalize_imagenet: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.freeze_dav2 = bool(freeze_dav2)
        self.in_channels = input_channels
        self.fc_output_channels = fc_output_channels

        self.vis_temp = None

        self.fully_conv = FullyConv(in_channels=input_channels, out_channels=fc_output_channels)

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

        print("[FullyConvDav2] DAv2 checkpoint:", str(checkpoint_path))
        print("[FullyConvDav2] DAv2 encoder:", dav2_encoder)
        print("[FullyConvDav2] DAv2 frozen:", self.freeze_dav2)
        print("[FullyConvDav2] Device:", self.device)
        print("[FullyConvDav2] FullyConv output channels:", self.fc_output_channels)
        print("[FullyConvDav2] Input size (H,W):", (input_size_height, input_size_width))
        print("[FullyConvDav2] Normalize ImageNet:", normalize_imagenet)
        print("[FullyConvDav2] FullyConv input channels:", self.in_channels)


    def set_dav2_frozen(self, freeze: bool = True) -> None:
        self.freeze_dav2 = bool(freeze)
        for param in self.dav2.parameters():
            param.requires_grad = not self.freeze_dav2

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        assert events.dim() == 4 and events.shape[1] == self.in_channels, f"Expected events of shape (B,{self.in_channels},H,W)"

        events = events.to(self.device)
        # The fully_conv network outputs a 3-channel image-like tensor
        reconstructed_rgb = self.fully_conv(events)

        # If the output is 1-channel, repeat it to make it 3-channel for DAV2
        if self.fc_output_channels == 1:
            reconstructed_rgb = reconstructed_rgb.repeat(1, 3, 1, 1)

        # Store for visualization if needed
        self.vis_temp = reconstructed_rgb.detach().cpu()

        # Pass the reconstructed image to DAV2 to get depth
        depth = self.dav2(reconstructed_rgb)
            
        return depth


class NewFullyConvDav2(torch.nn.Module):
    """Pipeline: events -> FullyConv -> DAV2 inverse-depth -> metric depth.

    Mirrors `NewUNetDav2` by learning the inverse-depth offset internally, so
    train/eval configs should keep `inv_prediction` disabled.
    """

    def __init__(
        self,
        input_channels: int = 5,
        dav2_encoder: str = "vits",
        dav2_checkpoint: Optional[str] = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        freeze_dav2: bool = True,
        device: torch.device = None,
        fc_output_channels: int = 3,
        normalize_imagenet: bool = False,
        inv_depth_constant_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.freeze_dav2 = bool(freeze_dav2)
        self.in_channels = input_channels
        self.fc_output_channels = fc_output_channels

        self.vis_temp = None
        self.fully_conv = FullyConv(in_channels=input_channels, out_channels=fc_output_channels)
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

        print("[NewFullyConvDav2] DAv2 checkpoint:", str(checkpoint_path))
        print("[NewFullyConvDav2] DAv2 encoder:", dav2_encoder)
        print("[NewFullyConvDav2] DAv2 frozen:", self.freeze_dav2)
        print("[NewFullyConvDav2] Device:", self.device)
        print("[NewFullyConvDav2] FullyConv output channels:", self.fc_output_channels)
        print("[NewFullyConvDav2] Input size (H,W):", (input_size_height, input_size_width))
        print("[NewFullyConvDav2] Normalize ImageNet:", normalize_imagenet)
        print("[NewFullyConvDav2] FullyConv input channels:", self.in_channels)
        print("[NewFullyConvDav2] Inverse-depth offset init:", float(inv_depth_constant_init))

    def set_dav2_frozen(self, freeze: bool = True) -> None:
        self.freeze_dav2 = bool(freeze)
        for param in self.dav2.parameters():
            param.requires_grad = not self.freeze_dav2

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        assert events.dim() == 4 and events.shape[1] == self.in_channels, f"Expected events of shape (B,{self.in_channels},H,W)"

        events = events.to(self.device)
        reconstructed_rgb = self.fully_conv(events)

        if self.fc_output_channels == 1:
            reconstructed_rgb = reconstructed_rgb.repeat(1, 3, 1, 1)

        self.vis_temp = reconstructed_rgb.detach().cpu()

        inv_depth = self.dav2(reconstructed_rgb)
        inv_depth_constant = torch.clamp(self.inv_depth_constant, min=1e-6)
        depth = 1.0 / (inv_depth + inv_depth_constant)
        return depth
