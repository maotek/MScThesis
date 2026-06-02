#!/bin/bash
#SBATCH --job-name=eval_dsec_train_mvsec_unet_small3_rerun
#SBATCH --output=eval_dsec_train_mvsec_unet_small3_rerun_%j.out
#SBATCH --error=eval_dsec_train_mvsec_unet_small3_rerun_%j.err
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

checkpoint="train_output/train_mvsec_unet_small3_dav2_batch10_rerun/epoch_050.pt"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing checkpoint: ${checkpoint}"
  echo "Submit train_mvsec_unet_small3_dav2_batch10_rerun.sh before this evaluation job."
  exit 1
fi

apptainer exec --nv ../../apptainer/image.sif python -m evaluate --config-path configs/dsec/validation/train_mvsec_unet_small3_dav2_batch10_rerun.json
