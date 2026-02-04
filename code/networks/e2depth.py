from pathlib import Path
from typing import Optional
import sys

import torch

_E2DEPTH_ROOT = Path(__file__).resolve().parents[1] / "models" / "rpg_e2depth" / "e2depth"
if str(_E2DEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_E2DEPTH_ROOT))

from models.rpg_e2depth.e2depth.model.model import *


def _default_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class E2Depth(torch.nn.Module):
    """Thin wrapper around the pretrained RPG E2Depth model.

    Expects voxel-grid event tensors shaped (B, C, H, W) where C matches the
    checkpoint's `num_bins`. Returns depth predictions shaped (B, 1, H, W).
    Maintains recurrent state if the checkpoint is recurrent; call `reset_state()`
    between sequences.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = _default_device(device)
        if weights_path is None:
            repo_root = Path(__file__).resolve().parents[1] / "models" / "rpg_e2depth"
            self.weights_path = str(repo_root / "pretrained" / "E2DEPTH_si_grad_loss_mixed.pth.tar")
        else:
            self.weights_path = weights_path

        # Load using the repo utility, forcing CPU to avoid CUDA dependency, then move to target device
        raw_model = torch.load(self.weights_path, map_location="cpu")
        arch = raw_model["arch"]
        try:
            model_type = raw_model["model"]
        except KeyError:
            model_type = raw_model["config"]["model"]

        model = eval(arch)(model_type)
        model.load_state_dict(raw_model["state_dict"])

        self.model = model.to(self.device)
        self.model.eval()

        # Determine if the loaded model is recurrent by inspecting attributes
        self.is_recurrent = hasattr(self.model, "unetrecurrent") or "Recurrent" in self.model.__class__.__name__

        print(f"Loading E2Depth checkpoint from {self.weights_path} (recurrent={self.is_recurrent})")
        print("Device:", self.device)

        self._states = None

    def reset_state(self) -> None:
        self._states = None

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if events.dim() == 3:
            events = events.unsqueeze(0)
        assert events.dim() == 4, "events must be (B,C,H,W)"

        self.reset_state()

        events = events.to(self.device)

        # For now, always pass None as states to avoid issues with state management
        result = self.model(events, None)

        if isinstance(result, tuple):
            pred, states = result
        else:
            pred, states = result, None

        if self.is_recurrent:
            self._states = states

        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        return pred


def load_e2depth(weights_path: Optional[str] = None, device: Optional[torch.device] = None) -> E2Depth:
    return E2Depth(weights_path=weights_path, device=device)
