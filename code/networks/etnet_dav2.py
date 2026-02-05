from typing import Optional

import torch

from networks.etnet import ETNet
from networks.dav2 import Dav2


class ETNetDav2(torch.nn.Module):
    """Pipeline: events -> ET-Net intensity -> DAV2 depth.

    Expects voxel-grid events (B,C,H,W) matching ET-Net input.
    ET-Net outputs (B,1,H,W) or (B,3,H,W); we ensure 3 channels and feed DAV2 for depth.
    """

    def __init__(
        self,
        etnet_checkpoint: str = None,
        dav2_encoder: str = "vits",
        dav2_checkpoint: str = None,
        device: torch.device = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        use_minmax_norm: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.etnet = ETNet(checkpoint_path=etnet_checkpoint, device=self.device, use_minmax_norm=use_minmax_norm)
        self.dav2 = Dav2(encoder=dav2_encoder, checkpoint=dav2_checkpoint, device=self.device, input_size_width=input_size_width, input_size_height=input_size_height)
        self.to(self.device)

    def reset_state(self) -> None:
        self.etnet.reset_state()

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: (B,C,H,W)
        intensity = self.etnet(events)  # (B,1,H,W) or (B,3,H,W)
        if intensity.shape[1] == 1:
            intensity_3ch = intensity.repeat(1, 3, 1, 1)  # (B,3,H,W)
        else:
            intensity_3ch = intensity
        depth = self.dav2(intensity_3ch)
        return depth
