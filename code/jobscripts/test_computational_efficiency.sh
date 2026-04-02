#!/bin/bash
#SBATCH --job-name=test_comp_eff
#SBATCH --output=test_comp_eff_%j.out
#SBATCH --error=test_comp_eff_%j.err
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
    --output-path output/test_computational_efficiency/summary.md
