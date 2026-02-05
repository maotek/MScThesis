from typing import Optional

import torch

from networks.e2vid import E2VID
from networks.dav2 import Dav2


class E2VIDDav2(torch.nn.Module):
    """Pipeline: events -> E2VID intensity -> DAV2 depth.

    Expects voxel-grid events (B,C,H,W) matching E2VID num_bins (default 5 for lightweight checkpoint).
    E2VID outputs (B,1,H,W); we replicate to 3 channels and feed DAV2 for depth.
    """

    def __init__(
        self,
        e2vid_weights: str = None,
        dav2_encoder: str = "vits",
        dav2_checkpoint: str = None,
        input_size_width: int = 350,
        input_size_height: int = 266,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.e2vid = E2VID(weights_path=e2vid_weights, device=self.device)
        self.dav2 = Dav2(encoder=dav2_encoder, checkpoint=dav2_checkpoint, device=self.device, input_size_width=input_size_width, input_size_height=input_size_height)
        self.to(self.device)

    def reset_state(self) -> None:
        self.e2vid.reset_state()

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: (B,C,H,W)
        intensity = self.e2vid(events)  # (B,1,H,W)
        intensity_3ch = intensity.repeat(1, 3, 1, 1)  # (B,3,H,W)
        depth = self.dav2(intensity_3ch)
        return depth
