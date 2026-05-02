from pathlib import Path
from typing import Optional

import torch

from networks.dae import DAE
from networks.fully_conv import FullyConv


class FullyConvDAE(torch.nn.Module):
    """Pipeline: events -> FullyConv -> Depth AnyEvent depth."""

    def __init__(
        self,
        input_channels: int = 5,
        dae_encoder: str = "vits",
        dae_checkpoint: Optional[str] = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        freeze_dae: bool = True,
        device: torch.device = None,
        fc_output_channels: int = 3,
        dae_input_channels: int = 3,
        dae_activation: str = "relu",
        dae_scale_factor: float = 1.0,
        dae_inv_prediction: bool = True,
        freeze_encoder: bool = False,
        dae_nopretrain: bool = False,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.freeze_dae = bool(freeze_dae)
        self.in_channels = input_channels
        self.fc_output_channels = fc_output_channels
        self.dae_input_channels = dae_input_channels
        self.apply_sigmoid = bool(apply_sigmoid)

        self.vis_temp = None
        self.fully_conv = FullyConv(in_channels=input_channels, out_channels=fc_output_channels)

        if dae_checkpoint is None:
            dae_checkpoint = str(
                Path(__file__).resolve().parents[1]
                / "models"
                / "depthanyevent"
                / "weights"
                / "dav2"
                / "finetuned_dsec"
                / "finetuned_dsec.pth"
            )

        self.dae = DAE(
            encoder=dae_encoder,
            checkpoint=str(Path(dae_checkpoint)),
            device=self.device,
            input_size_width=input_size_width,
            input_size_height=input_size_height,
            activation=dae_activation,
            scale_factor=dae_scale_factor,
            inv_prediction=dae_inv_prediction,
            freeze_encoder=freeze_encoder,
            input_channels=dae_input_channels,
            nopretrain=dae_nopretrain,
        )
        if self.freeze_dae:
            self.set_dae_frozen(True)

        if self.device is not None:
            self.to(self.device)

        print("[FullyConvDAE] DAE checkpoint:", str(Path(dae_checkpoint)))
        print("[FullyConvDAE] DAE encoder:", dae_encoder)
        print("[FullyConvDAE] DAE frozen:", self.freeze_dae)
        print("[FullyConvDAE] Device:", self.device)
        print("[FullyConvDAE] FullyConv output channels:", self.fc_output_channels)
        print("[FullyConvDAE] DAE input channels:", self.dae_input_channels)
        print("[FullyConvDAE] Input size (H,W):", (input_size_height, input_size_width))
        print("[FullyConvDAE] Apply sigmoid:", self.apply_sigmoid)

    def set_dae_frozen(self, freeze: bool = True) -> None:
        self.freeze_dae = bool(freeze)
        for param in self.dae.parameters():
            param.requires_grad = not self.freeze_dae

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        assert events.dim() == 4 and events.shape[1] == self.in_channels, f"Expected events of shape (B,{self.in_channels},H,W)"

        if self.device is not None:
            events = events.to(self.device)
        dae_input = self.fully_conv(events)
        if self.apply_sigmoid:
            dae_input = torch.sigmoid(dae_input)

        if self.fc_output_channels == 1 and self.dae_input_channels == 3:
            dae_input = dae_input.repeat(1, 3, 1, 1)
        if dae_input.shape[1] != self.dae_input_channels:
            raise ValueError(
                f"FullyConv output has {dae_input.shape[1]} channels, but DAE expects {self.dae_input_channels}"
            )

        self.vis_temp = dae_input.detach().cpu()
        return self.dae.model.infer_image(dae_input)
