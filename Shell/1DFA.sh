#!/bin/bash
#SBATCH --job-name=1DFA-L40-rnn-group1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 Main1D.py --L 40 --Model '1DFA' --net 'rnn' --max_stepAll 2000 --max_stepLater 100 --lr 0.001 --net_depth 3 --net_width 128 --print_step 100 --batch_size 1000 --Tstep 10001 --dlambda 0.1 --dlambdaL -3 --dlambdaR -2.75 --delta_t 0.1 --c 0.2 --cuda 0 --dtype float64

#exit 0

