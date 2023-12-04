# Transition rule for Voter model

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
from gru2DFirstUp import GRU2D
from lstm2D import LSTM2D
from made import MADE
from made1D import MADE1D
from mdrnn import RNN2D
from mdtensorizedrnn import MDTensorizedRNN
from pixelcnnFirstUp import PixelCNN
from stacked_pixelcnnFA import StackedPixelCNN
from utils import (
    clear_checkpoint,
    clear_log,
    default_dtype_torch,
    ensure_dir,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


plt.rc('font', size=16)


def TransitionState(sample, args, Tstep, step, net_new):
    Sample1D = (sample.view(-1, args.size) + 1) / 2  # sample has size batchsize X systemSize
    # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
    SampleNeighbor1D1 = Sample1D.repeat(args.size, 1, 1).permute(1, 0, 2)
    SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
    Mask = torch.eye(args.size).expand(Sample1D.shape[0], args.size, args.size).to(args.device)
    SampleNeighbor1DExtend = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(1, 0, 2)
    # Sample1D=Sample1D[:,1:]#L^3 to L^3-1 neighbor by fixing the first spin up
    Sample2D = (sample.view(-1, args.L, args.L) + 1) / 2  # sample has size batchsize X L X L

    if args.BC == 1:
        # print('Zero BC.')
        Col1 = torch.cat(
            (torch.zeros(Sample2D.shape[0], args.L, 1).to(args.device), Sample2D[:, :, :-1]), 2
        )  # down sites
        Col2 = torch.cat((Sample2D[:, :, 1:], torch.zeros(Sample2D.shape[0], args.L, 1).to(args.device)), 2)
        Row1 = torch.cat((torch.zeros(Sample2D.shape[0], 1, args.L).to(args.device), Sample2D[:, :-1, :]), 1)
        Row2 = torch.cat((Sample2D[:, 1:, :], torch.zeros(Sample2D.shape[0], 1, args.L).to(args.device)), 1)
    if args.BC == 2:
        # print('Periodic BC.'): checked correct for L=3
        Col1 = torch.cat(
            (Sample2D[:, :, -1].view(Sample2D.shape[0], args.L, 1), Sample2D[:, :, :-1]), 2
        )  # down sites
        Col2 = torch.cat((Sample2D[:, :, 1:], Sample2D[:, :, 0].view(Sample2D.shape[0], args.L, 1)), 2)
        Row1 = torch.cat((Sample2D[:, -1, :].view(Sample2D.shape[0], 1, args.L), Sample2D[:, :-1, :]), 1)
        Row2 = torch.cat((Sample2D[:, 1:, :], Sample2D[:, 0, :].view(Sample2D.shape[0], 1, args.L)), 1)
    if args.BC == 3:
        # print('Random BC.')
        Col1 = torch.cat(
            (torch.randint(0, 2, (Sample2D.shape[0], args.L, 1)).to(args.device), Sample2D[:, :, :-1]), 2
        )  # down sites
        Col2 = torch.cat(
            (Sample2D[:, :, 1:], torch.randint(0, 2, (Sample2D.shape[0], args.L, 1)).to(args.device)), 2
        )
        Row1 = torch.cat(
            (torch.randint(0, 2, (Sample2D.shape[0], 1, args.L)).to(args.device), Sample2D[:, :-1, :]), 1
        )
        Row2 = torch.cat(
            (Sample2D[:, 1:, :], torch.randint(0, 2, (Sample2D.shape[0], 1, args.L)).to(args.device)), 1
        )
    Wout = (
        torch.abs(Col1 - Sample2D)
        + torch.abs(Col2 - Sample2D)
        + torch.abs(Row1 - Sample2D)
        + torch.abs(Row2 - Sample2D)
    ).view(-1, args.size) / 4
    # BatchSize: The escape-probability for each sampled state to all connected states#torch.sum((Sample1D-1).abs()*args.c+Sample1D*(1-args.c),1)
    R = torch.sum(Wout, 1)

    SampleNeighbor2DExtend = SampleNeighbor1DExtend.view(-1, args.batch_size, args.L, args.L)
    # NeighborSize X BatchSize X L X L
    if args.BC == 1:
        # print('Zero BC.')
        Col1Neighbor = torch.cat(
            (
                torch.zeros(SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], args.L, 1).to(
                    args.device
                ),
                SampleNeighbor2DExtend[:, :, :, :-1],
            ),
            3,
        )  # down sites
        Col2Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, :, 1:],
                torch.zeros(SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], args.L, 1).to(
                    args.device
                ),
            ),
            3,
        )
        Row1Neighbor = torch.cat(
            (
                torch.zeros(SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], 1, args.L).to(
                    args.device
                ),
                SampleNeighbor2DExtend[:, :, :-1, :],
            ),
            2,
        )
        Row2Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, 1:, :],
                torch.zeros(SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], 1, args.L).to(
                    args.device
                ),
            ),
            2,
        )
    if args.BC == 2:
        # print('Periodic BC.'): checked correct for L=3
        Col1Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, :, -1].view(
                    SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], args.L, 1
                ),
                SampleNeighbor2DExtend[:, :, :, :-1],
            ),
            3,
        )  # down sites
        Col2Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, :, 1:],
                SampleNeighbor2DExtend[:, :, :, 0].view(
                    SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], args.L, 1
                ),
            ),
            3,
        )
        Row1Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, -1, :].view(
                    SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], 1, args.L
                ),
                SampleNeighbor2DExtend[:, :, :-1, :],
            ),
            2,
        )
        Row2Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, 1:, :],
                SampleNeighbor2DExtend[:, :, 0, :].view(
                    SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], 1, args.L
                ),
            ),
            2,
        )
    if args.BC == 3:
        # print('Random BC.')
        Col1Neighbor = torch.cat(
            (
                torch.randint(
                    0, 2, (SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], args.L, 1)
                ).to(args.device),
                SampleNeighbor2DExtend[:, :, :, :-1],
            ),
            3,
        )  # down sites
        Col2Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, :, 1:],
                torch.randint(
                    0, 2, (SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], args.L, 1)
                ).to(args.device),
            ),
            3,
        )
        Row1Neighbor = torch.cat(
            (
                torch.randint(
                    0, 2, (SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], 1, args.L)
                ).to(args.device),
                SampleNeighbor2DExtend[:, :, :-1, :],
            ),
            2,
        )
        Row2Neighbor = torch.cat(
            (
                SampleNeighbor2DExtend[:, :, 1:, :],
                torch.randint(
                    0, 2, (SampleNeighbor2DExtend.shape[0], SampleNeighbor2DExtend.shape[1], 1, args.L)
                ).to(args.device),
            ),
            2,
        )
    # Win has flipping rate of the previous state into these sampled states: NeighborSize X BatchSize X SystemSize
    WinTemp = (
        torch.abs(Col1Neighbor - SampleNeighbor2DExtend)
        + torch.abs(Col2Neighbor - SampleNeighbor2DExtend)
        + torch.abs(Row1Neighbor - SampleNeighbor2DExtend)
        + torch.abs(Row2Neighbor - SampleNeighbor2DExtend)
    ).view(SampleNeighbor2DExtend.shape[0], -1, args.size) / 4
    # Need to take i-th in the dimension3 for the i-th neighbor in dimension1:
    Win = torch.diagonal(WinTemp, offset=0, dim1=0, dim2=2)  # BatchSize X NeighborSize
    Win_lambda = Win  # No tilt for now
    # Checked: Win is right if WinTemp is right: the diagnal action is right.

    # New code 2: use logexpsum:
    with torch.no_grad():
        ratio = -torch.as_tensor(args.size * np.log(2), dtype=torch.float64).to(args.device)
        # Checked: correct for L=3
        if Tstep == 0:
            # ones*torch.log(c)+(args.size-1-ones)*torch.log(1-c)#BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
            LogP_t = torch.ones(args.batch_size).to(args.device) * ratio
            # ones.t()*torch.log(c)+(args.size-1-ones).t()*torch.log(1-c)# BatchSize X NeighborSize
            LogP_t_other = torch.ones(args.batch_size, args.size).to(args.device) * ratio
        else:
            LogP_t = net_new.log_prob(sample).detach()
            # Temp=torch.transpose(SampleNeighbor1D, 0, 1)#.view(sample.shape[0], args.size, args.size) #BatchSize X NeighborSize X SystemSize
            Temp = torch.transpose(SampleNeighbor1DExtend, 0, 1)
            # After transpose: BatchSize X NeighborSize X SystemSize

            Temp = Temp + (Temp - 1)  # Change 0 to -1 back: 1,-1 input to VAN
            if args.net == 'rnn' or args.net == 'rnn2' or args.net == 'lstm' or args.net == 'rnn3':
                Temp3 = torch.reshape(Temp, (args.batch_size * args.size, args.L, args.L))  # For RNN
            else:
                Temp3 = torch.reshape(Temp, (args.batch_size * args.size, 1, args.L, args.L))  # For VAN
            # BatchSize X NeighborSize: checked, it is consistent with for loop
            LogP_t_other = torch.reshape(net_new.log_prob(Temp3), (args.batch_size, args.size)).detach()
        Temp2 = (
            1
            + (torch.sum(torch.exp(LogP_t_other - LogP_t.repeat(args.size, 1).t()) * Win_lambda, 1) - R)
            * args.delta_t
        )
        if torch.min(Temp2) < 0:
            print('reduce delta t at ', step)
            Temp2[Temp2 <= 0] = 1e-300
        LogTP_t1 = torch.log(Temp2) + LogP_t

    return LogTP_t1


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1).float()


def AllTransitionState1D(args, dict2, Tstep, P_tnew):
    Win_lambda = torch.as_tensor(
        dict2['Win'] * np.float64(np.exp(-args.lambda_tilt)), dtype=torch.float64
    ).to(args.device)
    P_t = P_tnew.to(args.device)  # torch.exp(net_new.log_prob(sample)).detach()
    P_t_other = (
        torch.cat([P_tnew[dict2['IdConnec'][:, i].numpy()] for i in range(args.size - 1)])
        .to(args.device)
        .view(args.size - 1, -1)
        .t()
    )  # Correct: BatchSize X NeighborSize: checked, it is consistent with for loop
    TP_Exact = (
        P_t + (torch.sum(P_t_other * Win_lambda[:, :], 1) - dict2['R'].to(args.device) * P_t) * args.delta_t
    )

    return TP_Exact


def OptimizeFunction(net, params, optimizer, scheduler, net_new, args, lambda_tilt, Tstep):
    SampleT = []
    free_energy_mean3Temp = []
    Loss1Temp = []
    Loss1_2Temp = []
    ListDistanceCheck_Eucli = []
    ListDistanceCheck = []
    Listloss_mean = []
    Listloss_std = []
    for step in range(0, args.max_step + 1):
        optimizer.zero_grad()
        with torch.no_grad():
            sample, x_hat = net.sample(args.batch_size)
        log_prob = net.log_prob(sample)  # sample has size batchsize X 1 X systemSize
        with torch.no_grad():
            LogTP_t = TransitionState(sample, args, Tstep, step, net_new)
            TP_t_normalize = (
                torch.exp(LogTP_t) / (torch.exp(LogTP_t)).sum() * (torch.exp(log_prob)).sum()
            ).detach()
            loss = log_prob - LogTP_t.detach()
            lossL2 = torch.exp(log_prob) - TP_t_normalize
            losshe = -torch.sqrt(torch.exp(log_prob) * TP_t_normalize)

        assert not LogTP_t.requires_grad
        if args.lossType == 'kl':
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        elif args.lossType == 'klreweight':
            loss3 = torch.exp(log_prob) * (loss) / torch.exp(log_prob).mean()
            loss_reinforce = torch.mean((loss3 - loss3.mean()) * log_prob)
        elif args.lossType == 'l2':
            loss_reinforce = torch.mean((lossL2 - lossL2.mean()) * log_prob)
        elif args.lossType == 'he':
            loss_reinforce = torch.mean((losshe) * log_prob)
        elif args.lossType == 'ss':
            # steady state# Conversion from probability P to state \psi
            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob / 2)
        loss_reinforce.backward()

        if args.clip_grad:
            nn.utils.clip_grad_norm_(params, args.clip_grad)
        optimizer.step()
        if args.lr_schedule:
            scheduler.step(loss.mean())

        loss_std = loss.std()  # /args.size
        loss_mean = loss.mean()  # / args.size#(P_tnew * (P_tnew / (TP_t/torch.sum(TP_t))).log()).sum()
        Listloss_mean.append(loss_mean.detach().cpu().numpy())
        Listloss_std.append(loss_std.detach().cpu().numpy())
        DistanceCheck_Eucli = torch.sqrt(torch.sum((torch.exp(net.log_prob(sample)) - TP_t_normalize) ** 2))
        # function kl_div is not the same as wiki's explanation.
        DistanceCheck = torch.nn.functional.kl_div(net.log_prob(sample), TP_t_normalize, None, None, 'sum')
        ListDistanceCheck_Eucli.append(DistanceCheck_Eucli.detach().cpu().numpy())
        ListDistanceCheck.append(DistanceCheck.detach().cpu().numpy())
        if step > int(args.max_step * (1 - args.Percent)):
            with torch.no_grad():
                # loss.mean() #/ args.beta / args.size
                free_energy_mean3Temp.append(
                    torch.mean(torch.exp(log_prob) * (loss) / torch.exp(log_prob).mean())
                )
                Loss1Temp.append(loss_mean)
                Loss1_2Temp.append(loss_std)
        if step > int(args.max_step - 2):  # The number of last few epochs to record samples
            with torch.no_grad():
                SampleT.append(np.array(sample.detach().cpu()))
        if args.print_step and step % args.print_step == 0 and Tstep % int(args.print_step) == 0:
            my_log('Training...')
            my_log(
                'lambda={}, Time step of equation={}, Training step = {}, loss_reinforce = {:.20f}, loss_std={:.20f},loss_mean={}'.format(  # ',DynPartiFuncFactorLog={:.8f}'#' F = {:.8g}, F_std = {:.8g}, S = {:.8g}, E = {:.8g}, M = {:.8g}, Q = {:.8g}, lr = {:.3g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                    lambda_tilt,
                    Tstep,
                    step,
                    torch.abs(loss_reinforce),
                    torch.abs(loss_std),
                    torch.abs(loss_mean),  # DynPartiFuncFactorLog,
                )
            )
        ####################
    return (
        net,
        optimizer,
        SampleT,
        free_energy_mean3Temp,
        Loss1Temp,
        Loss1_2Temp,
        ListDistanceCheck_Eucli,
        DistanceCheck_Eucli,
        ListDistanceCheck,
        Listloss_mean,
        Listloss_std,
    )  # return TP_Exact


def Optimizer(net, args):
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    return optimizer, params, nparams
