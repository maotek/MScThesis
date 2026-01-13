import abc
import numpy as np

class EventRepresentation(abc.ABC):
    @abc.abstractmethod
    def from_events(self, events: np.ndarray, shape: tuple) -> np.ndarray:
        """Convert events to a representation of shape (H, W) or (H, W, C)."""
        pass
