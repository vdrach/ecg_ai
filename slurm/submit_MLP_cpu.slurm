#!/bin/bash -l
#SBATCH -J AI
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH -p cpu

source  /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

EXE=train_MLP.py

python ../scripts/${EXE} --epochs 10 --device cpu --dataclass fast



















