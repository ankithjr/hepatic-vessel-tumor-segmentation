#!/bin/bash
#SBATCH --job-name=cldice_train
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=cldice_train_%j.out
#SBATCH --error=cldice_train_%j.err

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
module load cuDNN/9.10.2.21-CUDA-12.6.0

source /mnt/vstor/courses/csds463/arj94/nnunet_env/bin/activate

export nnUNet_raw=/mnt/vstor/courses/csds463/arj94/nnUNet_raw
export nnUNet_preprocessed=/mnt/vstor/courses/csds463/arj94/nnUNet_preprocessed
export nnUNet_results=/mnt/vstor/courses/csds463/arj94/nnUNet_results

export nnUNet_extTrainer=/mnt/vstor/courses/csds463/arj94/project1/custom_trainers

export nnUNet_compile=false

nnUNetv2_train 8 3d_fullres 0 -tr nnUNetTrainerClDice
