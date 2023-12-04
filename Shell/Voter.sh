#!/bin/bash
#SBATCH --job-name=Voter-RNN-1-16-L6-deltat005-BC3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH -o out/%j.out
#SBATCH -e out/%j.out
#SBATCH -t 13-2:00:00

# load the environment
module purge

CUDA_LAUNCH_BLOCKING=1 python3 MainVoter.py --L 6 --Model 'Voter' --net 'rnn' --lossType 'kl' --max_stepAll 2000 --max_stepLater 30 --lr 0.001 --net_depth 1 --net_width 16 --print_step 100 --batch_size 100 --Tstep 20001 --dlambda 0.1 --dlambdaL -1.7 --dlambdaR -1.66 --delta_t 0.05 --c 0.5 --cuda 0 --dtype float64 --negativeS --BC 3

#exit 0

