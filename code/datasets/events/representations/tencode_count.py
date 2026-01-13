import numpy as np
from .event_representation import EventRepresentation

class TencodeCount(EventRepresentation):
    def __init__(self, timewindow: float):
        """Tencode-style frame with polarity, temporal gradient, and normalized event count."""
        self.timewindow = timewindow

    def from_events(self, events: dict, shape: tuple) -> np.ndarray:
        """Return a (H, W, 3) uint8 frame.
        R: polarity (255 for positive, 0 for negative)
        G: temporal gradient (as in Tencode)
        B: normalized event count in the window
        """
        H, W = shape
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        if len(events["t"]) == 0:
            return frame

        xs = events["x"]
        ys = events["y"]
        ts = events["t"]
        ps = events["p"]

        t_max = ts.max()
        g = (t_max - ts) / self.timewindow
        g = np.clip(g, 0.0, 1.0)
        g = (255.0 * g).astype(np.uint8)

        count_map = np.zeros((H, W), dtype=np.int32)
        np.add.at(count_map, (ys, xs), 1)
        if count_map.max() > 0:
            count_norm = (count_map.astype(np.float32) / count_map.max()) * 255.0
        else:
            count_norm = np.zeros_like(count_map, dtype=np.float32)
        count_norm = count_norm.astype(np.uint8)

        for x, y, p, g_val in zip(xs, ys, ps, g):
            frame[y, x, 0] = 255 if p > 0 else 127
            frame[y, x, 1] = g_val

        frame[..., 2] = count_norm

        return frame
