# Transition rule for 1D KCM


import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
from numpy import sqrt
from torch import nn

from args import args
from bernoulli import BernoulliMixture
from gru import GRU
from made import MADE
from made1D import MADE1D
from pixelcnn import PixelCNN
from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rc('font', size=16)
torch.set_printoptions(precision=80)
# torch.cuda.set_device(0)


def TransitionState(sample, args, Tstep, net_new):
    Sample1D = (sample.view(-1, args.size) + 1) / 2  # sample has size batchsize X systemSize
    Win = (Sample1D - 1).abs() * (
        1 - args.c
    ) + Sample1D * args.c  # The previous state flip into these sampled states
    if args.Model == '1DFA':
        fNeighbor = torch.cat(
            (torch.ones(Sample1D.shape[0], 1).to(args.device), Sample1D[:, :-1]), 1
        ) + torch.cat((Sample1D[:, 1:], torch.ones(Sample1D.shape[0], 1).to(args.device)), 1)
    if args.Model == '1DEast':
        fNeighbor = torch.cat((torch.ones(Sample1D.shape[0], 1).to(args.device), Sample1D[:, :-1]), 1)
    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size).expand(Sample1D.shape[0], args.size, args.size).to(args.device)
    SampleNeighbor1D = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)

    # BatchSize: The escape-probability for each sampled state to all connected states#torch.sum((Sample1D-1).abs()*args.c+Sample1D*(1-args.c),1)
    R = torch.sum((1 - Win) * fNeighbor, 1)
    Win = (
        Win * fNeighbor
    )  # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
    Win_lambda = torch.tensor(Win * np.exp(-args.lambda_tilt), dtype=torch.float64).to(args.device)
    # Extract the elements for  the previous-step state:  for the samples, we need to get the probility in the state vector:
    # For initial steady-state, just count 1s 0s. For later steps, use index to sample for VAN
    if args.Hermitian:
        WinHermitian = (Sample1D - 1).abs() * np.sqrt(args.c * (1 - args.c)) + Sample1D * np.sqrt(
            args.c * (1 - args.c)
        )  # The previous state flip into these sampled states
        # BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
        WinHermitian = WinHermitian * fNeighbor
        Win_lambda = torch.tensor(
            WinHermitian * np.float64(np.exp(-args.lambda_tilt)), dtype=torch.float64
        ).to(args.device)

    if Tstep == 0:
        with torch.no_grad():
            ones = torch.sum((Sample1D == 1.0), 1)
            # BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
            P_t = args.c**ones * (1 - args.c) ** (args.size - ones)
            ones = (SampleNeighbor1D == 1.0).sum(dim=2)
            P_t_other = (args.c**ones * (1 - args.c) ** (args.size - ones)).t()  # BatchSize X NeighborSize
    else:
        with torch.no_grad():
            P_t = torch.exp(net_new.log_prob(sample)).detach()
            # Temp=torch.transpose(SampleNeighbor1D, 0, 1).view(args.batch_size, args.size, args.size) #BatchSize X NeighborSize X SystemSize
            Temp = torch.transpose(SampleNeighbor1D, 0, 1).view(
                sample.shape[0], args.size, args.size
            )  # BatchSize X NeighborSize X SystemSize
            Temp = Temp + (Temp - 1)  # Change 0 to -1 back
            if args.net == 'rnn':
                Temp2 = torch.reshape(Temp, (args.batch_size * args.size, args.size))  # For RNN
            else:
                Temp2 = torch.reshape(Temp, (args.batch_size * args.size, 1, args.size))  # For VAN
            P_t_otherTemp = torch.exp(net_new.log_prob(Temp2)).detach()
            P_t_other = torch.reshape(P_t_otherTemp, (args.batch_size, args.size))
            # P_t_other=torch.exp(net_new.log_prob2(Temp)).detach()#BatchSize X NeighborSize: checked, it is consistent with for loop

    with torch.no_grad():
        TP_t = P_t + (torch.sum(P_t_other * Win_lambda, 1) - R * P_t) * args.delta_t
    return TP_t


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1).float()


def AllTransitionState1D(args, Tstep, P_tnew):
    if Tstep == 0:
        with torch.no_grad():
            # ones = torch.sum(torch.sum((StateAll== 1),1),1)
            P_t = args.P_ss.to(args.device)
            if args.Hermitian == True:
                P_t = P_t * args.PInverse12.to(args.device)
    else:
        P_t = P_tnew.to(
            args.device
        )  # After the first iteration, the last-time step probability is the learned VAN
    # Construct the transition matrix: only the matrix elements with hamming distance 1 of state are nonzero
    TP_t = torch.zeros(2 ** (args.size))  # TP_t:=T*P_t
    State = gen_all_binary_vectors(
        args.size
    )  # Configuration of spins: from left to right, up to down for the lattice
    # HammingD_Matrix=torch.cdist(State, State, p=0)#[0,1]
    Dhamming = (torch.cdist(State, State, p=0) == 1).nonzero(
        as_tuple=False
    )  # hamming distance by p=0 ##.nonzero(as_tuple=True)

    # Calculate each element of TP_t.
    for i in range(2 ** (args.size)):  # i-th state
        IdConnec = Dhamming[Dhamming[:, 0] == i, 1]  # The index of "transmissible" state to the i-th state

        # The transition probability depends on the current and next states' spin: The first is to get flip up/down, and the second is to know the flipped neighbor in the current state
        # Flip up or down from i-state to next, up is 1, down is 0
        UpDown = State[IdConnec][State[i] - State[IdConnec] != 0]
        IdFliped = torch.nonzero([State[i] - State[IdConnec] != 0][0].int())[
            :, 1
        ]  # The position of the flipped spin
        fNeighbor = torch.zeros_like(UpDown)  # checked: correct.
        for j in range(args.size):  # go through all the connected states
            if IdFliped[j] == 0:
                fNeighbor[j] = 1 + State[i, :][IdFliped[j] + 1]
            else:
                if args.Model == '1DFA':
                    if IdFliped[j] == args.size - 1:
                        fNeighbor[j] = 1 + State[i, :][IdFliped[j] - 1]
                    else:
                        fNeighbor[j] = State[i, :][IdFliped[j] - 1] + State[i, :][IdFliped[j] + 1]
                elif args.Model == '1DEast':
                    if IdFliped[j] == args.size - 1:
                        fNeighbor[j] = State[i, :][IdFliped[j] - 1]
                    else:
                        fNeighbor[j] = State[i, :][IdFliped[j] - 1]
        # Win=UpDown*args.c+(1-UpDown)*(1-args.c)
        Win = UpDown * (1 - args.c) + (1 - UpDown) * args.c
        # We have checked that fNeighbor has the same order of Win
        Wout = (
            1 - Win
        )  # The symmetric pairs of the nonzeor off-diagonal elements add up to 1, even after multipliying the neighboring effect
        if args.Model == '1DFA' or args.Model == '1DEast':
            Win = Win * fNeighbor
            Wout = Wout * fNeighbor  # Have the same fNeighbor after getting Wout by 1-Win
            if args.Hermitian == True:
                # BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
                Win = torch.tensor(args.PInverse12[i] * Win * args.P12[IdConnec], dtype=torch.float64).to(
                    args.device
                )
                # BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
                Wout = torch.tensor(args.PInverse12[IdConnec] * Wout * args.P12[i], dtype=torch.float64).to(
                    args.device
                )
        R = sum(Wout)  # Escape rate
        Win_lambda = torch.tensor(Win * np.exp(-args.lambda_tilt), dtype=torch.float64).to(args.device)
        # We get the elements of T*P_t, which is 2^L-elements vector:
        TP_t[i] = P_t[i] + (sum(P_t[IdConnec] * Win_lambda) - R.to(args.device) * P_t[i]) * args.delta_t
    return TP_t
