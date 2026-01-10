import torch
from torch.utils.data import Dataset
import numpy as np
import hdf5plugin
import h5py
import cv2
import os
from typing import Dict, List, Optional, Sequence, Tuple


# Import Tencode from representations
from representations.tencode import Tencode
# Import EventSlicer from DSEC
from datasets.DSEC.scripts.utils.eventslicer import EventSlicer

class TencodeDataset(Dataset):
    """Standalone PyTorch Dataset that returns Tencode RGB inputs and disparity targets.

    Input: tensor shape (3, H, W), dtype float32 (0-1)
    Target: tensor shape (1, H, W), dtype float32
    """
    def __init__(
        self,
        data_root: str,
        time_window_us: int = 10000,
        event_representation: Optional[Tencode] = None,
        shape: Tuple[int, int] = (480, 640),
        scenes: Optional[Sequence[str]] = None,
    ):
        self.data_root = data_root
        self.time_window_us = time_window_us
        self.shape = shape
        self.scenes_filter = set(scenes) if scenes is not None else None
        self.event_representation = event_representation or Tencode(timewindow=self.time_window_us)

        self.scene_metadata: Dict[str, Dict[str, object]] = {}
        self.samples: List[Dict[str, object]] = []
        self._build_index()

    def _build_index(self) -> None:
        for entry in sorted(os.listdir(self.data_root)):
            scene_path = os.path.join(self.data_root, entry)
            if not os.path.isdir(scene_path) or entry.startswith('.'):
                continue
            if self.scenes_filter is not None and entry not in self.scenes_filter:
                continue

            h5_path = os.path.join(scene_path, f"{entry}_events_left", "events.h5")
            disp_dir = os.path.join(scene_path, f"{entry}_disparity_event")
            timestamps_path = os.path.join(scene_path, "disparity_timestamps.txt")

            if not (os.path.exists(h5_path) and os.path.isdir(disp_dir) and os.path.exists(timestamps_path)):
                continue

            disp_files = sorted([f for f in os.listdir(disp_dir) if f.endswith('.png')])
            timestamps = np.loadtxt(timestamps_path, dtype=np.int64)
            if len(disp_files) != len(timestamps):
                raise ValueError(
                    f"Scene {entry} has {len(disp_files)} disparity files but {len(timestamps)} timestamps"
                )

            h5f = h5py.File(h5_path, 'r')
            self.scene_metadata[entry] = {
                "event_slicer": EventSlicer(h5f),
                "h5_file": h5f,
                "disp_dir": disp_dir,
                "timestamps": timestamps,
            }

            for i, disp_file in enumerate(disp_files):
                self.samples.append({
                    "scene": entry,
                    "disp_file": disp_file,
                    "timestamp": timestamps[i],
                })

        if not self.samples:
            raise ValueError(f"No valid scenes found in {self.data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        meta = self.scene_metadata[sample["scene"]]

        disp_ts = int(sample["timestamp"])
        slicer = meta["event_slicer"]

        # Clamp window to non-negative (respect slicer offset) and ensure at least 1us span
        t_min = slicer.get_start_time_us()
        t_start_us = max(disp_ts - self.time_window_us, t_min)
        t_end_us = max(disp_ts, t_start_us + 1)

        events = slicer.get_events(t_start_us, t_end_us)
        if events is None:
            events = {"p": np.empty((0,), dtype=np.uint8), "x": np.empty((0,), dtype=np.int16), "y": np.empty((0,), dtype=np.int16), "t": np.empty((0,), dtype=np.int64)}

        tencode_frame = self.event_representation.from_events(events, self.shape)

        disp_path = os.path.join(meta["disp_dir"], sample["disp_file"])
        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
        if disp_img is not None:
            disp_img = disp_img.astype(np.float32) / 256.0
        else:
            disp_img = np.zeros(self.shape, dtype=np.float32)
        x = torch.from_numpy(tencode_frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(disp_img.astype(np.float32)).unsqueeze(0)
        return x, y

    def __del__(self) -> None:
        for meta in self.scene_metadata.values():
            h5_file = meta.get("h5_file")
            try:
                h5_file.close()  # type: ignore[attr-defined]
            except Exception:
                pass
