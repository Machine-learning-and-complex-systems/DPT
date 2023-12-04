#!/bin/bash
#SBATCH --job-name=2DFANegativeS-rnn-L6-c005-group3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-22:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 Main2D.py --L 6 --Model '2DFA' --net 'rnn' --lossType 'kl' --max_stepAll 2000 --max_stepLater 100 --lr 0.001 --net_depth 1 --net_width 128 --print_step 100 --batch_size 1000 --Tstep 2003 --dlambda 0.2 --dlambdaL -0.01 --dlambdaR 0.1 --delta_t 0.05 --c 0.05 --cuda 0 --dtype float64 --Hermitian --negativeS

#exit 0

