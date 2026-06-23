#!/bin/bash -l
#SBATCH -J scaling_h100
#SBATCH -N 1
#SBATCH -n 1                       # one task; torchrun launches per-GPU processes
#SBATCH -c 16                      # cores per task -- 16 per GPU on a 64-core/4-GPU node
#SBATCH -t 02:00:00
#SBATCH -p gpu_h100
#SBATCH --gres=gpu:4               # all 4 H100s on the node
#SBATCH -o scaling_h100_%j.out

# This template runs ONE configuration; for the full scaling sweep,
# use snakemake (see Snakefile) which will fan this out into one
# sbatch per (GPU_COUNT, model_size) combination.

source /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

# Override these from the command line or environment per-job
SCRIPT="${SCRIPT:-../scripts/train_resnet1d.py}"
NPROC="${NPROC:-4}"                # use the existing value of NPROC if it's already set in the environment; otherwise default to 4- use e.g NPROC=2 sbatch scaling_h100.slurm            #to use 2 GPU
EXTRA_ARGS="${EXTRA_ARGS:---base_width 64 --depth 2}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
RESULT_CSV="${RESULT_CSV:-results_h100.csv}"

echo "Running ${SCRIPT} with NPROC=${NPROC} ${EXTRA_ARGS}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    ${SCRIPT} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --tf32 --mixed_precision \
    --result_csv ${RESULT_CSV} \
    ${EXTRA_ARGS}
