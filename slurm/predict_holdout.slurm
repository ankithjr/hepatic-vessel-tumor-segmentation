#!/bin/bash
#SBATCH -A csds463
#SBATCH -p markov_gpu
#SBATCH --job-name=nnunet_predict
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/predict_%j.out
#SBATCH --error=slurm/predict_%j.err

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
module load cuDNN/9.10.2.21-CUDA-12.6.0

source /mnt/vstor/courses/csds463/arj94/nnunet_env/bin/activate

export nnUNet_raw=/mnt/vstor/courses/csds463/arj94/nnUNet_raw
export nnUNet_preprocessed=/mnt/vstor/courses/csds463/arj94/nnUNet_preprocessed
export nnUNet_results=/mnt/vstor/courses/csds463/arj94/nnUNet_results

nnUNetv2_predict \
-i /mnt/vstor/courses/csds463/arj94/project1/holdout/imagesTs \
-o /mnt/vstor/courses/csds463/arj94/project1/holdout_predictions/baseline \
-d 8 \
-c 3d_fullres \
-f 0 \
-chk checkpoint_final.pth \
-device cuda \
--disable_progress_bar
