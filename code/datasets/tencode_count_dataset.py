import os
from typing import Optional, Sequence, Tuple

from datasets.tencode_dataset import TencodeDataset
from representations.tencode_count import TencodeCount


class TencodeCountDataset(TencodeDataset):
    """Dataset variant that defaults to the TencodeCount representation."""

    def __init__(
        self,
        data_root: str,
        time_window_us: int = 10000,
        shape: Tuple[int, int] = (480, 640),
        scenes: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            data_root=data_root,
            time_window_us=time_window_us,
            event_representation=TencodeCount(timewindow=time_window_us),
            shape=shape,
            scenes=scenes,
        )
