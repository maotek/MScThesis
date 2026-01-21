import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from models.depthanyevent.models.dav2 import get_depth_anything_v2


class DAE(torch.nn.Module):
    """Thin wrapper around Depth AnyEvent's DAv2 model for easy loading/inference.

    Args:
        encoder: one of vits/vitb/vitl/vitg.
        checkpoint: optional checkpoint path; defaults to models/depthanyevent/checkpoints/finetuned_dsec.pth.
        device: torch device or None to auto-select cuda/mps/cpu.
        input_size: (width, height) resize applied before forwarding into the model.
        activation: output activation used by Depth AnyEvent (relu/sigmoid/softplus).
        scale_factor: scale factor applied to the depth prediction.
        inv_prediction: whether to invert depth predictions.
        freeze_encoder: freeze the ViT encoder weights.
        input_channels: number of input channels expected by the model.
        nopretrain: if True, skip loading the checkpoint weights.
    """

    def __init__(
        self,
        encoder: str = "vits",
        checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
        input_size: int = 518,
        activation: str = "softplus",
        scale_factor: float = 1.0,
        inv_prediction: bool = True,
        freeze_encoder: bool = False,
        input_channels: int = 3,
        nopretrain: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.device = self._select_device(device)

        ckpt = checkpoint or os.path.join(
            Path(__file__).resolve().parent.parent,
            "models",
            "depthanyevent",
            "checkpoints",
            "finetuned_dsec.pth",
        )
        if not nopretrain and not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        print("Loading DepthAnyEvent checkpoint from:", ckpt)
        print("Using encoder:", encoder)
        print("Input size:", input_size)
        print("Device:", self.device)

        self.model = get_depth_anything_v2(
            checkpoint_path=ckpt,
            encoder=encoder,
            activation=activation,
            scale_factor=scale_factor,
            inv_prediction=inv_prediction,
            input_size_width=input_size,
            input_size_height=input_size,
            freeze_encoder=freeze_encoder,
            input_channels=input_channels,
            nopretrain=nopretrain,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _select_device(device: Optional[torch.device]) -> torch.device:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B,3,H,W) tensor in [0,1].
        Returns:
            depth: (B,1,H,W) resized back to input spatial size.
        """
        assert x.dim() == 4 and x.shape[1] == 3, "Expected x of shape (B,3,H,W)"
        orig_hw = x.shape[-2:]
        x_resized = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        depth, _ = self.model(x_resized)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.shape[1] != 1:
            depth = depth[:, :1]
        depth = F.interpolate(depth, size=orig_hw, mode="bilinear", align_corners=False)
        return depth

    def to_device(self, device: torch.device) -> "DAE":
        self.device = device
        self.model.to(device)
        return self
