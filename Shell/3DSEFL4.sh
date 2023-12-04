#!/bin/bash
#SBATCH --job-name=3DSouthEastBack-L4-made3-128-group3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 Main3D.py --L 4 --Model '3DSouthEastBack' --net 'made3d' --lossType 'kl' --max_stepAll 5000 --max_stepLater 100 --lr 0.001 --net_depth 3 --net_width 128 --print_step 100 --batch_size 1000 --Tstep 2001 --dlambda 0.1 --dlambdaL -1 --dlambdaR -0.55 --delta_t 0.05 --c 0.5 --cuda 0 --dtype float64

#exit 0

