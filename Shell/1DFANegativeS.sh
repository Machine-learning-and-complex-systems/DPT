#!/bin/bash
#SBATCH --job-name=1DEast-L20-negativeS-c005
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 Main1D.py --L 20 --Model '1DEast' --net 'made' --max_stepAll 1000 --max_stepLater 40 --lr 0.002 --net_depth 3 --net_width 64 --print_step 100 --batch_size 1000 --Tstep 20001 --dlambda 0.2 --dlambdaL -3 --dlambdaR 0.1 --delta_t 0.1 --c 0.05 --cuda 0 --dtype float64 --negativeS

#exit 0

