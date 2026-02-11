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
python -m evaluate --config-path configs/validation/dav2_tencode.json --csv-path results/dav2_tencode.csv
```

### DAV2 + Tencode PixelCount
```bash
python -m evaluate --config-path configs/validation/dav2_tencode_pixelcount.json --csv-path results/dav2_tencode_pixelcount.csv
```

### DAE + Tencode
```bash
python -m evaluate --config-path configs/validation/dae_tencode.json --csv-path results/dae_tencode.csv
```

### DAV2 + RGB
```bash
python -m evaluate --config-path configs/validation/dav2_rgb.json --csv-path results/dav2_rgb.csv
```

### E2VID-DAV2 + VoxelGrid
```bash
python -m evaluate --config-path configs/validation/e2vid_dav2_voxelgrid.json --csv-path results/e2vid_dav2_voxelgrid.csv
```

### E2VID-DAV2 Composite + VoxelGrid
```bash
python -m evaluate --config-path configs/validation/e2vid_dav2_composite_voxelgrid.json --csv-path results/e2vid_dav2_composite_voxelgrid.csv
```

### ETNet-DAV2 + VoxelGrid
```bash
python -m evaluate --config-path configs/validation/etnet_dav2_voxelgrid.json --csv-path results/etnet_dav2_voxelgrid.csv
```

### Concentration + DAv2
```bash
python -m evaluate --config-path configs/validation/concentration_dav2_voxelgrid.json --csv-path results/concentration_dav2_voxelgrid.csv
```

## Notes

- Visualization settings (`vis_interval`, `vis_dir`) are controlled by each JSON config.
- `evaluate.py` automatically creates parent directories for `--csv-path`.
