# Event Depth Evaluation

This folder contains the evaluation pipeline for event-based depth models on DSEC.

## Quick Start

Run these commands from the `code/` directory.

### 1) Create environment
```bash
conda env create -f environment/environment.yml
```

### 2) Activate environment
```bash
conda activate apptainer
```

### 3) Download DSEC data
```bash
python -m scripts.download_dsec
```

## Run Evaluation

Each run uses a config from `configs/validation` and writes metrics to `results/`.

### DAV2 + Tencode
```bash
python -m evaluate --config-path configs/dsec/validation/dav2_tencode.json --csv-path results/dav2_tencode.csv
```

### DAV2 + Tencode PixelCount
```bash
python -m evaluate --config-path configs/dsec/validation/dav2_tencode_pixelcount.json --csv-path results/dav2_tencode_pixelcount.csv
```

### DAE + Tencode
```bash
python -m evaluate --config-path configs/dsec/validation/dae_tencode.json --csv-path results/dae_tencode.csv
```

Test on MVSEC using the MVSEC weights
```bash
python -m evaluate --config-path configs/mvsec/validation/dae_tencode.json --csv-path results/dae_tencode_MVSEC.csv
```

### DAV2 + RGB
```bash
python -m evaluate --config-path configs/dsec/validation/dav2_rgb.json --csv-path results/dav2_rgb.csv
```

### E2VID-DAV2 + VoxelGrid
```bash
python -m evaluate --config-path configs/dsec/validation/e2vid_dav2_voxelgrid.json --csv-path results/e2vid_dav2_voxelgrid.csv
```

### E2VID-DAV2 Composite + VoxelGrid
```bash
python -m evaluate --config-path configs/dsec/validation/e2vid_dav2_composite_voxelgrid.json --csv-path results/e2vid_dav2_composite_voxelgrid.csv
```

### ETNet-DAV2 + VoxelGrid
```bash
python -m evaluate --config-path configs/dsec/validation/etnet_dav2_voxelgrid.json --csv-path results/etnet_dav2_voxelgrid.csv
```

### UNet + DAv2
Evaluate the model
```bash
python -m evaluate --config-path configs/dsec/validation/unet_dav2.json --csv-path results/unet_dav2.csv
```

Train the model

```bash
python -m train_unet_dav2 --config configs/dsec/train/unet_dav2.json --save-dir output/train_unet_dav2
```

### UNet + DAv2 5 batch
Evaluate the model
```bash
python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch5.json --csv-path results/unet_dav2_batch5.csv
```

Train the model

```bash
python -m train_unet_dav2 --config configs/dsec/train/unet_dav2_batch5.json --save-dir output/train_unet_dav2_batch5
```

### UNet + DAv2 (on RGB)
Evaluate the model
```bash
python -m evaluate --config-path configs/dsec/validation/unet_dav2_rgb.json --csv-path results/unet_dav2_rgb.csv
```

Train the model

```bash
python -m train_unet_dav2_rgb --config configs/dsec/train/unet_dav2_rgb.json --save-dir output/train_unet_dav2_rgb
```

## Notes

- Visualization settings (`vis_interval`, `vis_dir`) are controlled by each JSON config.
- `evaluate.py` automatically creates parent directories for `--csv-path`.
