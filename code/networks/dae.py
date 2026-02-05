import os
from pathlib import Path
from typing import Optional

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
        checkpoint: str = None,
        device: torch.device = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        activation: str = "relu",
        scale_factor: float = 1.0,
        inv_prediction: bool = True,
        freeze_encoder: bool = False,
        input_channels: int = 3,
        nopretrain: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.input_size_width = input_size_width
        self.input_size_height = input_size_height
        self.device = device

        if not nopretrain and not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        
        print("[DAE] Loading DepthAnyEvent checkpoint from:", checkpoint)
        print("[DAE] Using encoder:", encoder)
        print("[DAE] Input size width:", input_size_width)
        print("[DAE] Input size height:", input_size_height)
        print("[DAE] Device:", self.device)

        self.model = get_depth_anything_v2(
            checkpoint_path=checkpoint,
            encoder=encoder,
            activation=activation,
            scale_factor=scale_factor,
            inv_prediction=inv_prediction,
            input_size_width=input_size_width,
            input_size_height=input_size_height,
            freeze_encoder=freeze_encoder,
            input_channels=input_channels,
            nopretrain=nopretrain,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B,3,H,W) tensor in [0,1].
        Returns:
            depth: (B,1,H,W) resized back to input spatial size.
        """
        assert x.dim() == 4 and x.shape[1] == 3, "Expected x of shape (B,3,H,W)"
        
        return self.model.infer_image(x)