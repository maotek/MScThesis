import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Ensure the RPG E2VID repo is on path
E2VID_ROOT = Path(__file__).resolve().parents[1] / "models" / "rpg_e2vid"
if str(E2VID_ROOT) not in sys.path:
    sys.path.insert(0, str(E2VID_ROOT))

from utils.loading_utils import load_model  # type: ignore


def _default_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class E2VIDWrapper(torch.nn.Module):
    """Thin wrapper around the pretrained RPG E2VID model.

    Expects voxel-grid event tensors shaped (B, C, H, W) where C matches the
    checkpoint's `num_bins` (lightweight pretrained uses 5). Returns intensity
    reconstructions in [0,1] shaped (B, 1, H, W). Maintains recurrent state if
    the checkpoint is recurrent; call `reset_state()` between sequences.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = _default_device(device)
        self.weights_path = weights_path or str(E2VID_ROOT / "pretrained" / "E2VID_lightweight.pth.tar")

        self.model = load_model(self.weights_path)
        self.model.to(self.device)
        self.model.eval()

        # Determine if the loaded model is recurrent by inspecting attributes
        self.is_recurrent = hasattr(self.model, "unetrecurrent") or "Recurrent" in self.model.__class__.__name__
        self._states = None

    def reset_state(self) -> None:
        self._states = None

    @torch.no_grad()
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if events.dim() == 3:
            events = events.unsqueeze(0)
        assert events.dim() == 4, "events must be (B,C,H,W)"

        events = events.to(self.device)
        result = self.model(events, self._states) if self.is_recurrent else self.model(events, None)

        if isinstance(result, tuple):
            pred, states = result
        else:
            pred, states = result, None

        if self.is_recurrent:
            self._states = states

        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        return pred


def load_e2vid(weights_path: Optional[str] = None, device: Optional[torch.device] = None) -> E2VIDWrapper:
    return E2VIDWrapper(weights_path=weights_path, device=device)
