#!/bin/bash
#SBATCH --job-name=1DEast-L40-deltat005
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 Main1D.py --L 40 --Model '1DEast' --net 'made' --max_stepAll 2000 --max_stepLater 100 --lr 0.002 --net_depth 3 --net_width 64 --print_step 100 --batch_size 1000 --Tstep 1001 --dlambda 0.1 --dlambdaL -1 --dlambdaR -0.91 --delta_t 0.05 --c 0.5 --cuda 0 --dtype float64

#exit 0

