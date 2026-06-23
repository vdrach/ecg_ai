#!/bin/bash -l
#SBATCH -J preprocess
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 01:00:00
#SBATCH -p cpu_shared
#####SBATCH -o DNA_out_100000000_N1

source  /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

python ../scripts/preprocess.py --data-dir /users/vdrach/ecg_ai/data/raw/ltafdb --output-dir /users/vdrach/ecg_ai/data/preprocessed &
TRAIN_PID=$!
python ../scripts/resource_monitor.py --interval 1 --out preprocess_${SLURM_JOB_PARTITION}.csv --pid ${TRAIN_PID} &
MONITOR_PID=$!
wait $TRAIN_PID
kill $MONITOR_PID 2>/dev/null   # then stop the monitor cleanly
wait $MONITOR_PID 2>/dev/null   # let it flush/exit

#mpirun -np $SLURM_NTASKS nsys profile python laplacian_gpu.py
