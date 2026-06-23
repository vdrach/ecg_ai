#!/bin/bash -l
#SBATCH -J snakemake_scaling_h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 03:00:00
#SBATCH -p gpu_l40s
#SBATCH --gres=gpu:8
#SBATCH -o snakemake_scaling_l40s_%j.out

source /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

# Snakemake's --cores 1 here means "one snakemake rule instance at a time"
# (i.e. sequential runs). Each rule INTERNALLY launches torchrun with up to
# 4 GPUs -- so we serialize the runs through Snakemake but use all the
# allocated GPUs in parallel within each individual rule.
snakemake -s Snakefile_scaling --cores 1 -p --config node=l40s
