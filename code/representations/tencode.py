import numpy as np
from .event_representation import EventRepresentation

class Tencode(EventRepresentation):
    def __init__(self, timewindow: float):
        """Tencode temporal window delta-t (same units as event timestamps)."""
        self.timewindow = timewindow

    def from_events(self, events: dict, shape: tuple) -> np.ndarray:
        """Return a (H, W, 3) uint8 Tencode frame."""
        H, W = shape
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        if len(events['t']) == 0:
            return frame

        xs = events['x']
        ys = events['y']
        ts = events['t']
        ps = events['p']

        # t_max = latest event timestamp in the window
        t_max = ts.max()

        # normalized temporal gradient (clipped to [0, 1])
        g = (t_max - ts) / self.timewindow
        g = np.clip(g, 0.0, 1.0)
        g = (255.0 * g).astype(np.uint8)

        for x, y, p, g_val in zip(xs, ys, ps, g):
            if p > 0:
                # positive polarity red channel
                frame[y, x, 0] = 255
                frame[y, x, 1] = g_val
                frame[y, x, 2] = 0
            else:
                # negative polarity blue channel
                frame[y, x, 0] = 0
                frame[y, x, 1] = g_val
                frame[y, x, 2] = 255

        return frame
