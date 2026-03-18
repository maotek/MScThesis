# Event Depth Evaluation

This folder contains training and evaluation pipelines for event-based depth models on DSEC and MVSEC.

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

## Commands By Config

### Train Configs

Only training commands are listed here.

| Config | Train |
|---|---|
| `configs/dsec/train/fully_conv_dav2_batch10.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/fully_conv_dav2_batch10.json --save-dir output/train_dsec_fully_conv_dav2_batch10` |
| `configs/dsec/train/unet_dav2.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2.json --save-dir output/train_dsec_unet_dav2` |
| `configs/dsec/train/unet_dav2_batch10.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch10.json --save-dir output/train_dsec_unet_dav2_batch10` |
| `configs/dsec/train/unet_dav2_batch10_ch16.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch10_ch16.json --save-dir output/train_dsec_unet_dav2_batch10_ch16` |
| `configs/dsec/train/unet_dav2_batch10_ch16_RC.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch10_ch16_RC.json --save-dir output/train_dsec_unet_dav2_batch10_ch16_RC` |
| `configs/dsec/train/unet_dav2_batch10_ch8.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch10_ch8.json --save-dir output/train_dsec_unet_dav2_batch10_ch8` |
| `configs/dsec/train/unet_dav2_batch20.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch20.json --save-dir output/train_dsec_unet_dav2_batch20` |
| `configs/dsec/train/unet_dav2_batch5.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch5.json --save-dir output/train_dsec_unet_dav2_batch5` |
| `configs/dsec/train/unet_dav2_rgb.json` | `python -m train_unet_dav2_rgb --config-path configs/dsec/train/unet_dav2_rgb.json --save-dir output/train_dsec_unet_dav2_rgb` |
| `configs/dsec/train/unet_small2_dav2_batch10_ch16.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_small2_dav2_batch10_ch16.json --save-dir output/train_dsec_unet_small2_dav2_batch10_ch16` |
| `configs/dsec/train/unet_small2_dav2_batch10_ch8.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_small2_dav2_batch10_ch8.json --save-dir output/train_dsec_unet_small2_dav2_batch10_ch8` |
| `configs/dsec/train/unet_small2_dav2_batch5_ch16.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_small2_dav2_batch5_ch16.json --save-dir output/train_dsec_unet_small2_dav2_batch5_ch16` |
| `configs/dsec/train/unet_small2_dav2_batch5_ch8.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_small2_dav2_batch5_ch8.json --save-dir output/train_dsec_unet_small2_dav2_batch5_ch8` |
| `configs/dsec/train/unet_small3_dav2_batch10.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_small3_dav2_batch10.json --save-dir output/train_dsec_unet_small3_dav2_batch10` |
| `configs/dsec/train/unet_small3_dav2_batch10_ch16.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_small3_dav2_batch10_ch16.json --save-dir output/train_dsec_unet_small3_dav2_batch10_ch16` |
| `configs/dsec/train/unet_small3_dav2_batch10_ch8.json` | `python -m train_unet_dav2 --config-path configs/dsec/train/unet_small3_dav2_batch10_ch8.json --save-dir output/train_dsec_unet_small3_dav2_batch10_ch8` |
| `configs/mvsec/train/unet_dav2_mvsec.json` | `python -m train_unet_dav2 --config-path configs/mvsec/train/unet_dav2.json --save-dir output/train_mvsec_unet_dav2` |

### Evaluation Configs

Only evaluation commands are listed here.

| Config | Evaluate |
|---|---|
| `configs/dsec/validation/dae_tencode.json` | `python -m evaluate --config-path configs/dsec/validation/dae_tencode.json --csv-path results/dsec_validation_dae_tencode.csv` |
| `configs/dsec/validation/dav2_rgb.json` | `python -m evaluate --config-path configs/dsec/validation/dav2_rgb.json --csv-path results/dsec_validation_dav2_rgb.csv` |
| `configs/dsec/validation/dav2_tencode.json` | `python -m evaluate --config-path configs/dsec/validation/dav2_tencode.json --csv-path results/dsec_validation_dav2_tencode.csv` |
| `configs/dsec/validation/dav2_tencode_pixelcount.json` | `N/A (config is commented-out JSON)` |
| `configs/dsec/validation/e2vid_dav2_composite_voxelgrid.json` | `N/A (config is commented-out JSON)` |
| `configs/dsec/validation/e2vid_dav2_voxelgrid.json` | `python -m evaluate --config-path configs/dsec/validation/e2vid_dav2_voxelgrid.json --csv-path results/dsec_validation_e2vid_dav2_voxelgrid.csv` |
| `configs/dsec/validation/etnet_dav2_voxelgrid.json` | `python -m evaluate --config-path configs/dsec/validation/etnet_dav2_voxelgrid.json --csv-path results/dsec_validation_etnet_dav2_voxelgrid.csv` |
| `configs/dsec/validation/unet_dav2.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2.json --csv-path results/dsec_validation_unet_dav2.csv` |
| `configs/dsec/validation/unet_dav2_batch10.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10.json --csv-path results/dsec_validation_unet_dav2_batch10.csv` |
| `configs/dsec/validation/unet_dav2_batch10_ch16_0.5grad.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16_0.5grad.json --csv-path results/dsec_validation_unet_dav2_batch10_ch16_0.5grad.csv` |
| `configs/dsec/validation/unet_dav2_batch10_ch16_100ep.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16_100ep.json --csv-path results/dsec_validation_unet_dav2_batch10_ch16_100ep.csv` |
| `configs/dsec/validation/unet_dav2_batch10_ch16.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16.json --csv-path results/dsec_validation_unet_dav2_batch10_ch16.csv` |
| `configs/dsec/validation/unet_dav2_batch10_ch16_RC.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16_RC.json --csv-path results/dsec_validation_unet_dav2_batch10_ch16_RC.csv` |
| `configs/dsec/validation/unet_dav2_batch10_ch16_nograd.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16_nograd.json --csv-path results/dsec_validation_unet_dav2_batch10_ch16_nograd.csv` |
| `configs/dsec/validation/unet_dav2_batch10_ch8.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch8.json --csv-path results/dsec_validation_unet_dav2_batch10_ch8.csv` |
| `configs/dsec/validation/unet_dav2_batch20.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch20.json --csv-path results/dsec_validation_unet_dav2_batch20.csv` |
| `configs/dsec/validation/unet_dav2_batch5.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch5.json --csv-path results/dsec_validation_unet_dav2_batch5.csv` |
| `configs/dsec/validation/unet_dav2_rgb.json` | `python -m evaluate --config-path configs/dsec/validation/unet_dav2_rgb.json --csv-path results/dsec_validation_unet_dav2_rgb.csv` |
| `configs/dsec/validation/unet_small2_dav2_batch10_ch16.json` | `python -m evaluate --config-path configs/dsec/validation/unet_small2_dav2_batch10_ch16.json --csv-path results/dsec_validation_unet_small2_dav2_batch10_ch16.csv` |
| `configs/dsec/validation/unet_small2_dav2_batch10_ch8.json` | `python -m evaluate --config-path configs/dsec/validation/unet_small2_dav2_batch10_ch8.json --csv-path results/dsec_validation_unet_small2_dav2_batch10_ch8.csv` |
| `configs/dsec/validation/unet_small2_dav2_batch5_ch16.json` | `python -m evaluate --config-path configs/dsec/validation/unet_small2_dav2_batch5_ch16.json --csv-path results/dsec_validation_unet_small2_dav2_batch5_ch16.csv` |
| `configs/dsec/validation/unet_small2_dav2_batch5_ch8.json` | `python -m evaluate --config-path configs/dsec/validation/unet_small2_dav2_batch5_ch8.json --csv-path results/dsec_validation_unet_small2_dav2_batch5_ch8.csv` |
| `configs/mvsec/validation/dae_tencode.json` | `python -m evaluate --config-path configs/mvsec/validation/dae_tencode.json --csv-path results/mvsec_validation_dae_tencode.csv` |
| `configs/mvsec/validation/dae_tencode_DSEC_checkpoint.json` | `python -m evaluate --config-path configs/mvsec/validation/dae_tencode_DSEC_checkpoint.json --csv-path results/mvsec_validation_dae_tencode_DSEC_checkpoint.csv` |
| `configs/mvsec/validation/e2vid_dav2_voxelgrid.json` | `python -m evaluate --config-path configs/mvsec/validation/e2vid_dav2_voxelgrid.json --csv-path results/mvsec_validation_e2vid_dav2_voxelgrid.csv` |
| `configs/mvsec/validation/unet_dav2.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2.json --csv-path results/mvsec_validation_unet_dav2.csv` |
| `configs/mvsec/validation/unet_dav2_batch10.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch10.json --csv-path results/mvsec_validation_unet_dav2_batch10.csv` |
| `configs/mvsec/validation/unet_dav2_batch10_ch16_0.5grad.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch10_ch16_0.5grad.json --csv-path results/mvsec_validation_unet_dav2_batch10_ch16_0.5grad.csv` |
| `configs/mvsec/validation/unet_dav2_batch10_ch16_100ep.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch10_ch16_100ep.json --csv-path results/mvsec_validation_unet_dav2_batch10_ch16_100ep.csv` |
| `configs/mvsec/validation/unet_dav2_batch10_ch16.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch10_ch16.json --csv-path results/mvsec_validation_unet_dav2_batch10_ch16.csv` |
| `configs/mvsec/validation/unet_dav2_batch10_ch16_RC.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch10_ch16_RC.json --csv-path results/mvsec_validation_unet_dav2_batch10_ch16_RC.csv` |
| `configs/mvsec/validation/unet_dav2_batch10_ch16_nograd.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch10_ch16_nograd.json --csv-path results/mvsec_validation_unet_dav2_batch10_ch16_nograd.csv` |
| `configs/mvsec/validation/unet_dav2_batch10_ch8.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch10_ch8.json --csv-path results/mvsec_validation_unet_dav2_batch10_ch8.csv` |
| `configs/mvsec/validation/unet_dav2_batch5.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_batch5.json --csv-path results/mvsec_validation_unet_dav2_batch5.csv` |
| `configs/mvsec/validation/unet_dav2_mvsec.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_dav2_mvsec.json --csv-path results/mvsec_validation_unet_dav2_mvsec.csv` |
| `configs/mvsec/validation/unet_small2_dav2_batch10_ch16.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_small2_dav2_batch10_ch16.json --csv-path results/mvsec_validation_unet_small2_dav2_batch10_ch16.csv` |
| `configs/mvsec/validation/unet_small2_dav2_batch10_ch8.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_small2_dav2_batch10_ch8.json --csv-path results/mvsec_validation_unet_small2_dav2_batch10_ch8.csv` |
| `configs/mvsec/validation/unet_small2_dav2_batch5_ch16.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_small2_dav2_batch5_ch16.json --csv-path results/mvsec_validation_unet_small2_dav2_batch5_ch16.csv` |
| `configs/mvsec/validation/unet_small2_dav2_batch5_ch8.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_small2_dav2_batch5_ch8.json --csv-path results/mvsec_validation_unet_small2_dav2_batch5_ch8.csv` |
| `configs/mvsec/validation/unet_small3_dav2_batch10_ch16.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_small3_dav2_batch10_ch16.json --csv-path results/mvsec_validation_unet_small3_dav2_batch10_ch16.csv` |
| `configs/mvsec/validation/unet_small3_dav2_batch10_ch8.json` | `python -m evaluate --config-path configs/mvsec/validation/unet_small3_dav2_batch10_ch8.json --csv-path results/mvsec_validation_unet_small3_dav2_batch10_ch8.csv` |

## Notes

- `evaluate.py` and both training scripts use `--config-path`.
- Two validation configs are fully commented out and not directly runnable as JSON:
  - `configs/dsec/validation/dav2_tencode_pixelcount.json`
  - `configs/dsec/validation/e2vid_dav2_composite_voxelgrid.json`
- `evaluate.py` creates parent directories for `--csv-path`; training scripts create `--save-dir`.
