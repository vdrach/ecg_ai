#!/bin/bash -l
#SBATCH -J AI
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH -p gpu_h100
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#####SBATCH -o DNA_out_100000000_N1

source  /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

EXE=train_MLP_exp.py

python ../scripts/${EXE} --epochs 10 --device cuda --dataclass fast 
#python ../scripts/${EXE} --epochs 10 --device cuda --dataclass fast  --model tiny --data_on_gpu


#python ../scripts/${EXE} --epochs 10 --device cuda --dataclass fast  --model tiny --data_on_gpu --manual_batching
#python ../scripts/${EXE}  --epochs 10 --device cuda --dataclass fast  --model big --data_on_gpu
#python ../scripts/${EXE} --epochs 10 --device cuda --dataclass fast  --model big --data_on_gpu --manual_batching

# Push batch size hard, with data_on_gpu so loader overhead is fully eliminated
#python ../scripts/${EXE} --device cuda --dataclass fast --data_on_gpu \
#    --batch_size 8192 --epochs 10 --model tiny

#python ../scripts/${EXE} --device cuda --dataclass fast --data_on_gpu \
#    --batch_size 8192 --epochs 10 --model big

#python ../scripts/${EXE}  --device cuda --dataclass fast --data_on_gpu \
#    --batch_size 8192 --epochs 10 --model tiny --manual_batching

#python ../scripts/${EXE} --device cuda --dataclass fast --data_on_gpu \
#    --batch_size 8192 --epochs 10 --model big --manual_batching
























