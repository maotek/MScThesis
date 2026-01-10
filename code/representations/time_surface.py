
import numpy as np
from .event_representation import EventRepresentation


class TimeSurface(EventRepresentation):
    def __init__(self, tau: float):
        """Time-surface encoder with exponential decay (tau in event time units)."""
        self.tau = tau

    def from_events(self, events: dict, shape: tuple) -> np.ndarray:
        """Return a time surface of shape (H, W) with values in [0, 1]."""
        H, W = shape
        surface = np.zeros((H, W), dtype=np.float32)

        if len(events['t']) == 0:
            return surface

        t_now = events['t'][-1]  # Last event time in window
        last_ts = np.full((H, W), -np.inf, dtype=np.float32)

        xs, ys, ts = events['x'], events['y'], events['t']
        last_ts[ys, xs] = ts  # Overwrites, keeps latest event per pixel

        delta_t = t_now - last_ts
        surface = np.exp(-delta_t / self.tau)
        surface[last_ts == -np.inf] = 0.0

        return surface
