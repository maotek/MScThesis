#!/bin/bash
#SBATCH --job-name=rgb_stats_dsec
#SBATCH --output=rgb_stats_dsec_%j.out
#SBATCH --error=rgb_stats_dsec_%j.err
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --mem=32000

module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load miniconda # Load conda

cd MScThesis/code

apptainer exec ../../apptainer/image.sif python -m test_dsec.compute_rgb_stats
