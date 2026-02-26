from .events_representations import (
    EventRepresentation,
    VoxelGrid,
    Histogram,
    Tencode,
    TencodePixelCount,
    ETNetVoxelGrid,
    E2vidVoxelGrid,
)
from typing import Dict, Any

def fetch_event_representation(config: Dict[str, Any]) -> EventRepresentation:
    """Create event representation object from config dict."""
    rep_type = config["representation_type"]
    height = config.get("height")
    width = config.get("width")

    if rep_type == "tencode" or rep_type == "rgb":
        # For RGB, we still use Tencode representation but has no use case, since we will only use the RGB from the dataloader.
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
