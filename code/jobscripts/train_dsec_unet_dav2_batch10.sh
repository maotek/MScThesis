#!/bin/bash
#SBATCH --job-name=train_dsec_unet_dav2_batch10
#SBATCH --output=train_dsec_unet_dav2_batch10_%j.out
#SBATCH --error=train_dsec_unet_dav2_batch10_%j.err
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=160000

module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load cuda cudnn miniconda # Load certain versions of cuda and cudnn

cd MScThesis/code

apptainer exec --nv ../../apptainer/image.sif python -m train_unet_dav2 --config configs/dsec/train/unet_dav2_batch10.json --save-dir output/train_dsec_unet_dav2_batch10
