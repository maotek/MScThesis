import os
from typing import Dict, List, Tuple, Type, Optional, Any
from pprint import pprint
from tqdm import tqdm
import torch
import h5py
import hdf5plugin
import random
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from .sbt.mvsec_sequence import MVSECSequence
from ..events import EventRepresentation, fetch_event_representation
from ..utils import Augmentator, fetch_preprocessing
from .constants import (
    MVSEC_HEIGHT,
    MVSEC_WIDTH, 
    MVSEC_TRAIN,
    MVSEC_TEST,
    MVSEC_VALIDATION,
    MVSEC_ALL_DATA_FOLDERS,
)


def worker_init_fn(worker_id: int) -> None:
    torch_seed = torch.randint(0, 2**30, (1,)).item()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)



def load_datasets(
    mvsec_path: str,
    data_split: str,
    event_representation: EventRepresentation,
    augmentator: Optional[Augmentator],
    load_images: bool = False,
    overfit: bool = False,
    sequence_window: int = 1,
    sequence_step: int = 1,
    self_supervised: bool = False,
    postfix: str = "",
    time_window_ms: int = 50,
) -> Dict[str, Dataset]:
    # Validate input parameters
    assert data_split in ["train", "validation", "test"], (
        f"Invalid data_split '{data_split}'. Must be one of: train, validation, test"
    )

    print(f"Loading MVSEC dataset from: {mvsec_path}")
    print(f"Data split: {data_split}, Load images: {load_images}")

    # Determine sequences to load based on data split
    if data_split == "train":
        data_folders = MVSEC_TRAIN
    elif data_split == "validation":
        data_folders = MVSEC_VALIDATION
    elif data_split == "test":
        data_folders = MVSEC_TEST
    else:
        raise Exception(f"Unrecognized data_split: {data_split}")

    # Load and validate dataset sequences
    datasets: Dict[str, Dataset] = {}

    for folder_name in data_folders.keys():
        # Create primary sequence dataset
        sequence_path = os.path.join(mvsec_path, folder_name)
        
        datasets[folder_name] = MVSECSequence(
            sequence_path=sequence_path,
            event_representation=event_representation,
            time_window_ms=time_window_ms,
            augmentator=augmentator,
            load_images=load_images,
            overfit=overfit,
            sequence_window=sequence_window,
            sequence_step=sequence_step,
            split=data_split,
            self_supervised=self_supervised,
            postfix=postfix,
        )

    return datasets


def fetch_dataloader(config_dataloader: Dict[str, Any], test: bool = False):
    """
    Build MVSEC DataLoader dict.
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
        mvsec_path=datapath,
        data_split=datasplit,
        event_representation=event_representation,
        augmentator=augmentator,
        load_images=load_images,
        sequence_window=sequence_window,
        sequence_step=sequence_step,
        self_supervised=self_supervised,
        postfix=postfix,
        time_window_ms=time_window_ms,
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
            pin_memory=False,
            shuffle=not test,
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=num_workers > 0,
        )
    return dataloaders
