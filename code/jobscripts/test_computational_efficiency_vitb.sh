#!/bin/bash
#SBATCH --job-name=test_comp_eff_vitb
#SBATCH --output=test_comp_eff_vitb_%j.out
#SBATCH --error=test_comp_eff_vitb_%j.err
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=160000

module use /opt/insy/modulefiles # Use DAIC INSY software collection
module load cuda cudnn miniconda # Load certain versions of cuda and cudnn

cd MScThesis/code

apptainer exec --nv ../../apptainer/image.sif \
  python -m test_dsec.test_computational_efficiency \
    --dav2-encoder vitb \
    --dav2-checkpoint models/dav2/checkpoints/depth_anything_v2_vitb.pth \
    --output-path test_dsec_output/test_computational_efficiency/summary_vitb.md
