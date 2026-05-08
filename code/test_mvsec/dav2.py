import json
from pathlib import Path

import numpy as np
import torch

from datasets.MVSEC.mvsec_dataset import fetch_dataloader
from networks.dav2 import Dav2
from networks.fully_conv import FullyConv
from util import save_depth_colormap_with_cbar, save_image


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "configs" / "mvsec" / "validation" / "train_mvsec_new_fully_conv_dav2_batch10_augmented.json"
DAV2_CHECKPOINT = BASE_DIR / "models" / "dav2" / "checkpoints" / "depth_anything_v2_vits.pth"
OUTPUT_DIR = BASE_DIR / "test_mvsec_output" / "test_dav2_hardcoded_image"

SEQUENCE_NAME = "test/outdoor_day1"
SAMPLE_INDEX = 0


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["data_loader"], config["model"], config


def resolve_model_checkpoint(model_config):
    checkpoint_path = Path(model_config["checkpoint_path"])
    if checkpoint_path.is_absolute():
        return checkpoint_path
    return BASE_DIR / checkpoint_path


def load_sample(data_loader_config):
    dataloaders = fetch_dataloader(data_loader_config, test=True)
    dataset = dataloaders[SEQUENCE_NAME].dataset
    sample = dataset[SAMPLE_INDEX]
    events = sample["depth_aligned_events"][0].unsqueeze(0)
    return events


def load_fully_conv(model_config, checkpoint_path, device):
    input_channels = int(model_config.get("input_channels", 5))
    output_channels = int(model_config.get("fc_output_channels", 3))

    model = FullyConv(in_channels=input_channels, out_channels=output_channels)
    model = model.to(device)
    model.eval()

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    fc_state = {}
    for key, value in state.items():
        if key.startswith("fully_conv."):
            fc_state[key.replace("fully_conv.", "")] = value

    if not fc_state:
        raise KeyError("No fully_conv.* weights found in checkpoint: {}".format(checkpoint_path))

    model.load_state_dict(fc_state, strict=True)
    return model, state, output_channels


def save_reconstruction_debug(reconstructed_rgb):
    recon = reconstructed_rgb.detach().cpu().squeeze(0).numpy()
    recon_hwc = np.transpose(recon, (1, 2, 0))

    np.save(OUTPUT_DIR / "vis_temp_raw.npy", recon)

    clipped = np.clip(recon_hwc, 0.0, 1.0)
    save_image(str(OUTPUT_DIR / "vis_temp_clipped.png"), clipped)

    recon_min = float(recon_hwc.min())
    recon_max = float(recon_hwc.max())
    if recon_max > recon_min:
        minmax = (recon_hwc - recon_min) / (recon_max - recon_min)
    else:
        minmax = np.zeros_like(recon_hwc)
    save_image(str(OUTPUT_DIR / "vis_temp_minmax.png"), minmax)

    negative_mask = (recon_hwc < 0).any(axis=2).astype(np.float32)
    save_image(str(OUTPUT_DIR / "vis_temp_negative_mask.png"), negative_mask, cmap="gray")

    total = recon.size
    negative = int((recon < 0).sum())
    above_one = int((recon > 1).sum())

    print("Reconstruction stats:")
    print("  min:", float(recon.min()))
    print("  max:", float(recon.max()))
    print("  mean:", float(recon.mean()))
    print("  median:", float(np.median(recon)))
    print("  negative values:", negative, "/", total, "({:.2f}%)".format(100.0 * negative / total))
    print("  >1 values:", above_one, "/", total, "({:.2f}%)".format(100.0 * above_one / total))


def load_inv_depth_constant(state, model_config):
    if "inv_depth_constant" in state:
        return float(state["inv_depth_constant"])
    return float(model_config.get("inv_depth_constant_init", 1.0))


@torch.no_grad()
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError("Config not found: {}".format(CONFIG_PATH))
    if not DAV2_CHECKPOINT.is_file():
        raise FileNotFoundError("DAv2 checkpoint not found: {}".format(DAV2_CHECKPOINT))

    data_loader_config, model_config, _ = load_config(CONFIG_PATH)
    model_checkpoint = resolve_model_checkpoint(model_config)
    if not model_checkpoint.is_file():
        raise FileNotFoundError("Model checkpoint not found: {}".format(model_checkpoint))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Config:", CONFIG_PATH)
    print("Model checkpoint:", model_checkpoint)
    print("DAv2 checkpoint:", DAV2_CHECKPOINT)
    print("Sequence:", SEQUENCE_NAME)
    print("Sample index:", SAMPLE_INDEX)
    print("Device:", device)

    events = load_sample(data_loader_config).to(device)
    print("Events shape:", tuple(events.shape))
    print("Events stats: min={}, max={}, mean={}, median={}".format(
        float(events.min()),
        float(events.max()),
        float(events.mean()),
        float(torch.median(events)),
    ))

    fully_conv, state, output_channels = load_fully_conv(model_config, model_checkpoint, device)
    reconstructed_rgb = fully_conv(events)
    if output_channels == 1:
        reconstructed_rgb = reconstructed_rgb.repeat(1, 3, 1, 1)

    save_reconstruction_debug(reconstructed_rgb)

    dav2 = Dav2(
        encoder=str(model_config.get("dav2_encoder", "vits")),
        checkpoint=str(DAV2_CHECKPOINT),
        device=device,
        input_size_height=int(model_config.get("input_size_height", 266)),
        input_size_width=int(model_config.get("input_size_width", 350)),
        normalize_imagenet=bool(model_config.get("normalize_imagenet", False)),
    )

    pred_inv = dav2(reconstructed_rgb).squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_inv = np.nan_to_num(pred_inv, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(OUTPUT_DIR / "dav2_inv_depth.npy", pred_inv)
    save_depth_colormap_with_cbar(
        str(OUTPUT_DIR / "dav2_inv_depth.png"),
        pred_inv,
        label="DAv2 inverse-depth",
    )

    print("Raw DAv2 output stats:")
    print("  min:", float(pred_inv.min()))
    print("  max:", float(pred_inv.max()))
    print("  mean:", float(pred_inv.mean()))
    print("  median:", float(np.median(pred_inv)))

    inv_depth_constant = load_inv_depth_constant(state, model_config)
    pred_depth = 1.0 / (pred_inv + inv_depth_constant)
    pred_depth = np.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(OUTPUT_DIR / "dav2_depth_converted.npy", pred_depth)
    save_depth_colormap_with_cbar(
        str(OUTPUT_DIR / "dav2_depth_converted.png"),
        pred_depth,
        label="Converted depth",
    )

    print("Using inverse-depth constant:", inv_depth_constant)
    print("Converted depth stats:")
    print("  min:", float(pred_depth.min()))
    print("  max:", float(pred_depth.max()))
    print("  mean:", float(pred_depth.mean()))
    print("  median:", float(np.median(pred_depth)))

    print("Saved raw fully-conv tensor to", OUTPUT_DIR / "vis_temp_raw.npy")
    print("Saved clipped fully-conv visualization to", OUTPUT_DIR / "vis_temp_clipped.png")
    print("Saved min-max fully-conv visualization to", OUTPUT_DIR / "vis_temp_minmax.png")
    print("Saved negative-mask visualization to", OUTPUT_DIR / "vis_temp_negative_mask.png")
    print("Saved raw DAv2 inverse-depth to", OUTPUT_DIR / "dav2_inv_depth.npy")
    print("Saved converted depth to", OUTPUT_DIR / "dav2_depth_converted.npy")


if __name__ == "__main__":
    main()
