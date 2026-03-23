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

## Run Training

All training settings live in the JSON configs under `configs/dsec/train/` and `configs/mvsec/train/`.

Examples:
```bash
python -m train_unet_dav2 --config-path configs/dsec/train/unet_dav2_batch10.json
python -m train_unet_dav2_rgb --config-path configs/dsec/train/unet_dav2_rgb.json
```

Training output is written to the `training.save_dir` defined in the config.

## Run Evaluation (Single Config)

All evaluation settings, including `csv_path`, live in the JSON configs under:
`configs/dsec/validation/` and `configs/mvsec/validation/`.

Example:
```bash
python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16.json
```

Optional override to swap only the checkpoint filename:
```bash
python -m evaluate --config-path configs/dsec/validation/unet_dav2_batch10_ch16.json   --checkpoint-file epoch_050.pt
```

## Run All Evaluations

```bash
bash run_all_validations.sh
```

`run_all_validations.sh` iterates all validation configs, skips ones whose `csv_path`
already exists, and continues on failures.

## Notes

- All paths for checkpoints, CSV output, and visualization are defined in the JSON configs.
- `evaluate.py` creates parent directories for `csv_path`.
