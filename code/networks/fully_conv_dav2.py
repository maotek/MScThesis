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
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.device = device
        self.freeze_dav2 = bool(freeze_dav2)
        self.in_channels = input_channels
        self.output_channels = output_channels

        self.vis_temp = None

        self.fully_conv = FullyConv(in_channels=input_channels, out_channels=output_channels)

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
            rgb=False,
        )

        self.set_dav2_frozen(self.freeze_dav2)

        self.to(self.device)

        print("[FullyConvDav2] DAv2 checkpoint:", str(checkpoint_path))
        print("[FullyConvDav2] DAv2 encoder:", dav2_encoder)
        print("[FullyConvDav2] DAv2 frozen:", self.freeze_dav2)
        print("[FullyConvDav2] Device:", self.device)

    def set_dav2_frozen(self, freeze: bool = True) -> None:
        self.freeze_dav2 = bool(freeze)
        for param in self.dav2.parameters():
            param.requires_grad = not self.freeze_dav2

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        assert events.dim() == 4 and events.shape[1] == self.in_channels, f"Expected events of shape (B,{self.in_channels},H,W)"

        events = events.to(self.device)
        # The fully_conv network outputs a 3-channel image-like tensor
        reconstructed_rgb = self.fully_conv(events)

        # Store for visualization if needed
        self.vis_temp = reconstructed_rgb.detach().cpu()

        # Pass the reconstructed image to DAV2 to get depth
        depth = self.dav2(reconstructed_rgb)
            
        return depth
