"""
DSEC dataset loader utilities (aligned with depthanyevent dataloader style).
"""

import os
import random
from typing import Any, Dict, Optional

from pprint import pprint

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DSEC_HEIGHT, DSEC_WIDTH
from .sbt.dsec_sequence import DsecSequence
from ..events import (
    E2vidVoxelGrid,
    ETNetVoxelGrid,
    EventRepresentation,
    Histogram,
    Tencode,
    TencodePixelCount,
    VoxelGrid,
)
from ..utils import Augmentator, fetch_preprocessing


def worker_init_fn(worker_id: int) -> None:
    torch_seed = torch.randint(0, 2**30, (1,)).item()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def fetch_event_representation(config: Dict[str, Any]) -> EventRepresentation:
    """Create event representation object from config dict."""
    rep_type = config["representation_type"]
    height = config.get("height", DSEC_HEIGHT)
    width = config.get("width", DSEC_WIDTH)

    if rep_type == "tencode" or rep_type == "rgb":
        return Tencode(
            height=height,
            width=width,
            normalize=config.get("normalize", True),
            white_frame=config.get("white_frame", False),
        )
    if rep_type == "tencode_pixelcount":
        return TencodePixelCount(
            height=height,
            width=width,
            normalize=config.get("normalize", True),
            white_frame=config.get("white_frame", False),
        )
    if rep_type == "voxelgrid":
        return VoxelGrid(
            channels=config.get("channels", 5),
            height=height,
            width=width,
            normalize=config.get("normalize", True),
        )
    if rep_type == "etnet_voxelgrid":
        return ETNetVoxelGrid(
            channels=config.get("channels", 5),
            height=height,
            width=width,
            combined_voxel_channels=config.get("combined_voxel_channels", True),
            temporal_bilinear=config.get("temporal_bilinear", True),
        )
    if rep_type == "e2vid_voxelgrid":
        return E2vidVoxelGrid(
            channels=config.get("channels", 5),
            height=height,
            width=width,
        )
    if rep_type == "histogram":
        return Histogram(
            height=height,
            width=width,
            remove_int_artifact=config.get("remove_int_artifact", False),
        )

    raise ValueError(f"Unknown event representation type: {rep_type}")


def load_datasets(
    dsec_path: str,
    data_split: str,
    time_window_ms: Optional[int],
    event_representation: EventRepresentation,
    augmentator: Optional[Augmentator],
    load_images: bool = False,
    sequence_window: int = 1,
    sequence_step: int = 1,
    self_supervised: bool = False,
    postfix: str = "",
) -> Dict[str, Dataset]:
    """
    Create one dataset per DSEC sequence directory (depthanyevent-compatible style).
    """

    time_window_ms = time_window_ms if time_window_ms is not None else 50
    split_path = dsec_path + os.path.sep + data_split

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split path does not exist: {split_path}")

    sequence_names = sorted(
        entry
        for entry in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, entry))
    )
    if not sequence_names:
        raise FileNotFoundError(f"No sequence folders found in split path: {split_path}")

    datasets: Dict[str, Dataset] = {}
    for sequence_name in sequence_names:
        sequence_path = os.path.join(split_path, sequence_name)
        datasets[sequence_name] = DsecSequence(
            sequence_path=sequence_path,
            event_representation=event_representation,
            augmentator=augmentator,
            load_images=load_images,
            sequence_window=sequence_window,
            sequence_step=sequence_step,
            self_supervised=self_supervised,
            postfix=postfix,
            time_window_ms=time_window_ms,
        )
    return datasets


def fetch_dataloader(config_dataloader: Dict[str, Any], test: bool = False):
    """
    Build DSEC DataLoader dict.
    """
    if "datapath" in config_dataloader:
        datapath = config_dataloader["datapath"]
    else:
        raise ValueError("No datapath provided")
    
    # print config
    print("Dataloader config:")
    pprint(config_dataloader)

    datasplit = config_dataloader["split"]
    time_window_ms = config_dataloader.get("time_window_ms", 50)

    batch_size = config_dataloader.get("batch_size", 1)
    num_workers = config_dataloader.get("num_workers", 1)
    load_images = config_dataloader.get("load_images", False)
    concatenate_sequences = config_dataloader.get("concatenate_sequences", False)
    sequence_window = config_dataloader.get("sequence_window", 1)
    sequence_step = config_dataloader.get("sequence_step", 1)
    self_supervised = config_dataloader.get("self_supervised", False)
    postfix = config_dataloader.get("postfix", "")

    augmentator = fetch_preprocessing(config_dataloader["preprocessing"])
    ev_config = config_dataloader["event_representation"].copy()
    event_representation = fetch_event_representation(ev_config)

    datasets = load_datasets(
        dsec_path=datapath,
        data_split=datasplit,
        time_window_ms=time_window_ms,
        event_representation=event_representation,
        augmentator=augmentator,
        load_images=load_images,
        sequence_window=sequence_window,
        sequence_step=sequence_step,
        self_supervised=self_supervised,
        postfix=postfix,
    )
    print(len(datasets), "sequences loaded.")
    for key in datasets:
        print(f"Dataset {key} has {len(datasets[key])} samples")

    if concatenate_sequences:
        datasets = {"concatenated": torch.utils.data.ConcatDataset(list(datasets.values()))}

    dataloaders = {}
    for seq in datasets:
        dataloaders[seq] = torch.utils.data.DataLoader(
            datasets[seq],
            batch_size=batch_size,
            pin_memory=True,
            shuffle=not test,
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=num_workers > 0,
        )
    return dataloaders
