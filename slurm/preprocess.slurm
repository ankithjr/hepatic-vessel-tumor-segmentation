#!/bin/bash

#SBATCH -A csds463

#SBATCH -p markov_cpu

#SBATCH --job-name=nnunet_preprocess

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=04:00:00

#SBATCH --output=slurm/preprocess_%j.out

#SBATCH --error=slurm/preprocess_%j.err



module load Python/3.11.5-GCCcore-13.2.0



source /mnt/vstor/courses/csds463/arj94/nnunet_env/bin/activate



export nnUNet_raw=/mnt/vstor/courses/csds463/arj94/nnUNet_raw

export nnUNet_preprocessed=/mnt/vstor/courses/csds463/arj94/nnUNet_preprocessed

export nnUNet_results=/mnt/vstor/courses/csds463/arj94/nnUNet_results



nnUNetv2_plan_and_preprocess -d 8 --no_pbar

