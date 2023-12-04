#!/bin/bash
#SBATCH --job-name=2DFA-pixelcnn3-32-L6-group2-IS10
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 Main2D.py --L 6 --Model '2DFA' --net 'pixelcnn' --lossType 'kl' --max_stepAll 4000 --max_stepLater 50 --lr 0.001 --net_depth 3 --net_width 32 --print_step 100 --batch_size 1000 --Tstep 4001 --dlambda 0.1 --dlambdaL -1.7 --dlambdaR -1.66 --delta_t 0.05 --c 0.5 --cuda 0 --dtype float64 --half_kernel_size 2 --max_stepT1 11 --max_stepLater0 100

#exit 0

