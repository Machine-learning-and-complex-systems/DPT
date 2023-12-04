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
from Transition1D import AllTransitionState1D, TransitionState, gen_all_binary_vectors
from pixelcnn import PixelCNN
from made1D import MADE1D
from made import MADE
from gru import GRU
from bernoulli import BernoulliMixture
from args import args
from torch import nn
from numpy import sqrt
import torch
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


plt.rc('font', size=16)
torch.set_printoptions(precision=80)


def Test():
    # #Initialize parameters: see args.py for a the help information on the parameters if needed
    # #System parameters:
    args.Model = '1DFA'  # type of models: 'No','FA', 'East', see args.py for a the help information
    args.L = 3  # Lattice size: 1D
    args.Tstep = 11  # Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
    args.delta_t = 0.1  # Time step length
    args.dlambda = (
        0.1  # The steplength from the left to the rigth boundary value of the count field  (lambda=s)
    )
    args.dlambdaL = -1  # left boundary value of the count field  (lambda=s)
    args.dlambdaR = -0.95  # Rigth boundary value of the count field  (lambda=s)
    args.c = 0.5  # 0.2#0.5 #Flip-up probability
    args.negativeS = True  # For the negative counting field (s<0) or not (s>0)
    # #Neural-network hyperparameters:
    args.net = 'made'  # 'rnn'#'lstm'#'rnn'Type of neural network in the VAN
    args.max_stepAll = 500  # 0 #The epoch for the 1st  time steps
    args.max_stepLater = 50  # 00 #The epoch at time steps except for the 1st step
    args.print_step = 1  # 0   # The time step size of print and save results
    args.net_depth = 2  # 3#3  # Depth of the neural network
    args.net_width = 2  # 128 # Width of the neural network
    args.batch_size = 100  # 1000 #batch size

    # ###################################################
    # args.Tstep=11#10001#100#100#101#100#1000#00#0#00#00#0#00#000#10#00 #Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
    # args.Model='1DFA' # 'No','FA', 'East', 1D or 2D
    # args.Hermitian=False # Do Hermitian transform or not
    # args.delta_t=0.1#0.1#0.02# Time step length
    # args.dlambda=0.2#0.14# 0.1# Step length of the tilted parameter (counting field) lambda
    # args.lambda_tilt=0.5#tilt parameter for the generating function
    # args.dlambdaL=-1
    # args.dlambdaR=-0.5
    # args.c=0.5#0.2#0.5 #Flip-up probability
    # args.net = 'made'
    # args.print_step=100
    # args.max_stepAll=3#500#2000
    # args.max_stepLater=5
    # args.max_step=args.max_stepAll#args.max_step=max_step
    # args.net_depth=2#3#3  # including output layer and not input data layer
    # args.net_width=2#16#16#64#4
    # args.L=3#10#10#16 # Lattice size: 1D
    # args.batch_size=100#00#1000#00#00#00

    # #Default parameters
    args.max_step = args.max_stepAll  # args.max_step=max_step
    args.lossType = 'kl'  # 'ss'# #type of loss
    args.clip_grad = 1  # clip gradient
    # args.Hermitian=False # Do Hermitian transform or not
    # args.lr_schedule=1#1, 2
    args.bias = True  # With bias or not in the neural network
    args.size = args.L  # the number of spin: 1D, doesnt' count the boundary spins
    args.epsilon = 0  # 1e-6/(2**args.size) # avoid 0 value in log below
    args.free_energy_batch_size = args.batch_size
    lambda_tilt_Range = 10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda))  # plot heatmap
    if args.negativeS:
        lambda_tilt_Range = -(10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda)))  # plot heatmap
    start_time2 = time.time()

    for delta_tt in np.arange(1):  # The result is not sensitively depend on time steplength so far
        start_time = time.time()
        init_out_dir()
        if args.clear_checkpoint:
            clear_checkpoint()
        last_step = get_last_checkpoint_step()
        print(last_step)
        if last_step >= 0:
            my_log('\nCheckpoint found: {}\n'.format(last_step))
        else:
            clear_log()

        SummaryListDynPartiFuncLog = []
        SummaryListDynPartiFuncLog2 = []
        SummaryListDynPartiFuncLog3 = []
        SummaryLoss1 = []
        SummaryLoss1_2 = []
        SummaryLoss2 = []
        net_new = []
        SummarySampleSum = []

        if args.size <= 10:
            StateAll = gen_all_binary_vectors(args.size).view(
                2**args.size, 1, args.size
            )  # all states of 1,0
            StateAll = 2 * StateAll - 1  # 1,0 to 1,-1
            args.ones = torch.sum(torch.sum((StateAll == 1), 1), 1)
            # BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
            args.P_ss = args.c**args.ones * (1 - args.c) ** (args.size - args.ones)
            args.PInverse12 = 1 / torch.sqrt(args.P_ss)
            args.P12 = torch.sqrt(args.P_ss)

        for lambda_tilt in lambda_tilt_Range:
            args.lambda_tilt = lambda_tilt
            args.max_step = args.max_stepAll
            # Initialize net and optimizer
            if args.net == 'made':
                net = MADE1D(**vars(args))
            elif args.net == 'pixelcnn':
                net = PixelCNN(**vars(args))
            elif args.net == 'rnn':
                net = GRU(**vars(args))
            elif args.net == 'bernoulli':
                net = BernoulliMixture(**vars(args))
            else:
                raise ValueError('Unknown net: {}'.format(args.net))
            net.to(args.device)
            my_log('{}\n'.format(net))
            params = list(net.parameters())
            params = list(filter(lambda p: p.requires_grad, params))
            nparams = int(sum([np.prod(p.shape) for p in params]))
            my_log('Total number of trainable parameters: {}'.format(nparams))
            named_params = list(net.named_parameters())
            optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
            print_args()
            TP_Exact = []
            ListDynPartiFuncLog2 = []
            ListDynPartiFuncLog3 = []
            DynPartiFuncLog2 = 0
            DynPartiFuncLog3 = 0
            ListDynPartiFuncLog2_2 = []
            ListDynPartiFuncLog2_3 = []
            DynPartiFuncLog2_2 = 0
            DynPartiFuncLog2_3 = 0
            Loss1 = []
            Loss1_2 = []
            Loss2 = []
            TP_ExactRecord = []
            SampleSum = []

            for Tstep in range(args.Tstep):  # Time step of the dynamical equation
                # Start
                init_time = time.time() - start_time
                sample_time = 0
                train_time = 0
                start_time = time.time()

                ListDistanceCheck_Eucli = []
                ListDistanceCheck = []
                Listloss_mean = []
                Listloss_std = []
                if Tstep >= 1:
                    args.max_step = args.max_stepLater
                if args.lr_schedule:
                    if args.lr_schedule_type == 1:
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            factor=0.5,
                            patience=int(args.max_step * args.Percent),
                            verbose=True,
                            threshold=1e-4,
                            min_lr=1e-5,
                        )
                    if args.lr_schedule_type == 2:
                        scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer, lr_lambda=lambda epoch: 1 / (epoch * 10 * args.lr + 1)
                        )
                    if args.lr_schedule_type == 3:
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            optimizer, 10 ** (-2 / args.max_step)
                        )  # lr_final = 0.01 * lr_init (1e-2 -> 1e-4)
                    if args.lr_schedule_type == 4:
                        scheduler = torch.optim.lr_scheduler.CyclicLR(
                            optimizer,
                            base_lr=1e-4,
                            max_lr=1e-2,
                            step_size_up=20,
                            step_size_down=20,
                            mode='exp_range',
                            gamma=0.999,
                            scale_fn=None,
                            scale_mode='cycle',
                            cycle_momentum=False,
                            last_epoch=-1,
                        )

                if args.size <= 10:  # compare with numerical exact result
                    TP_Exact = AllTransitionState1D(args, Tstep, TP_Exact)
                    TP_ExactRecord.append(TP_Exact.detach().cpu().numpy())

                # Train VAN
                free_energy_mean3Temp = []
                Loss1Temp = []
                Loss1_2Temp = []
                SampleT = []

                for step in range(last_step + 1, args.max_step + 1):
                    optimizer.zero_grad()
                    sample_start_time = time.time()
                    with torch.no_grad():
                        sample, x_hat = net.sample(args.batch_size)
                    sample_time += time.time() - sample_start_time
                    train_start_time = time.time()

                    log_prob = net.log_prob(sample)  # sample has size batchsize X 1 X systemSize

                    with torch.no_grad():
                        aa = TransitionState(sample, args, Tstep, net_new).detach()
                        TP_t = torch.abs(
                            aa
                        )  # +torch.min(aa)*(1e-8)#args.epsilon # avoid 0 value in log below
                        TP_t_normalize = (TP_t / TP_t.sum() * (torch.exp(log_prob)).sum()).detach()
                        loss = log_prob - torch.log(TP_t)

                    assert not TP_t.requires_grad
                    loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
                    loss_reinforce.backward()

                    if args.clip_grad:
                        nn.utils.clip_grad_norm_(params, args.clip_grad)
                    optimizer.step()
                    if args.lr_schedule:
                        scheduler.step(loss.mean())
                    train_time += time.time() - train_start_time

                    loss_std = loss.std()  # /args.size
                    loss_mean = (
                        loss.mean()
                    )  # / args.size#(P_tnew * (P_tnew / (TP_t/torch.sum(TP_t))).log()).sum()
                    Listloss_mean.append(loss_mean.detach().cpu().numpy())
                    Listloss_std.append(loss_std.detach().cpu().numpy())
                    DistanceCheck_Eucli = torch.sqrt(
                        torch.sum((torch.exp(net.log_prob(sample)) - TP_t_normalize) ** 2)
                    )
                    # function kl_div is not the same as wiki's explanation.
                    DistanceCheck = torch.nn.functional.kl_div(
                        net.log_prob(sample), TP_t_normalize, None, None, 'sum'
                    )
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
                    if step > int(args.max_step - 2):  # max(int(args.max_step*(1-0.02)),1):
                        with torch.no_grad():
                            SampleT.append(np.array(sample.detach().cpu()))

                    # print out:
                    if args.print_step and step % args.print_step == 0 and Tstep % int(args.print_step) == 0:
                        if step > 0:
                            sample_time /= args.print_step
                            train_time /= args.print_step
                        used_time = time.time() - start_time
                        my_log('init_time = {:.3f}'.format(init_time))
                        my_log('Training...')
                        my_log(
                            # ',DynPartiFuncFactorLog={:.8f}'#' F = {:.8g}, F_std = {:.8g}, S = {:.8g}, E = {:.8g}, M = {:.8g}, Q = {:.8g}, lr = {:.3g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                            'lambda={}, Time step of equation={}, Training step = {}, used_time = {:.3f}, loss_reinforce = {:.20f}, loss_std={:.20f},loss_mean={}'.format(
                                lambda_tilt,
                                Tstep,
                                step,
                                used_time,
                                step,
                                torch.abs(loss_reinforce),
                                torch.abs(loss_std),
                                torch.abs(loss_mean),  # DynPartiFuncFactorLog,
                            )
                        )
                        sample_time = 0
                        train_time = 0

                with torch.no_grad():
                    free_energy_mean3 = torch.mean(
                        torch.tensor(free_energy_mean3Temp, dtype=torch.float64).to(args.device)
                    )  # loss.mean() #/ args.beta / args.size
                    if Tstep == 1000 or Tstep == 3000 or Tstep == 5000 or Tstep == 8000 or Tstep == 15000:
                        PATH = args.out_filename + str(Tstep)
                        torch.save(net, PATH)
                    net_new = copy.deepcopy(net)  # net
                    net_new.requires_grad = False
                    DynPartiFuncLog2_3 += free_energy_mean3.detach().cpu().numpy()
                    ListDynPartiFuncLog2_3.append(DynPartiFuncLog2_3)
                    Loss1.append(
                        np.array(
                            torch.mean(torch.tensor(Loss1Temp, dtype=torch.float64).to(args.device))
                            .detach()
                            .cpu()
                        )
                    )
                    Loss1_2.append(
                        np.array(
                            torch.mean(torch.tensor(Loss1_2Temp, dtype=torch.float64).to(args.device))
                            .detach()
                            .cpu()
                        )
                    )
                    Loss2.append(np.array(DistanceCheck_Eucli.detach().cpu()))
                    # if Tstep % int(args.print_step*50)==0: #if Tstep % (1/args.delta_t)==0:
                    #         SampleSum.append(np.array(SampleT).reshape(-1,args.size))
                    SampleSum.append(np.mean(np.array(SampleT).reshape(-1, args.size), 0))

                    if args.size <= 10:
                        free_energy_mean3 = -torch.log(torch.sum(TP_Exact))
                        # Each time it has summed all the sequential partition functions
                        DynPartiFuncLog3 = free_energy_mean3.detach().cpu().numpy()
                        ListDynPartiFuncLog3.append(DynPartiFuncLog3)

                # #Plot training loss
                if Tstep <= 1 or Tstep == 1000 or Tstep == 5000:
                    plt.figure(num=None, dpi=300, edgecolor='k')
                    fig, axes = plt.subplots(2, 1)
                    fig.tight_layout()
                    ax = plt.subplot(2, 1, 1)
                    # ,, label = 'Relative mean square error')
                    plt.plot(range(0, args.max_step + 1), np.abs(Listloss_mean), label='Loss_mean')
                    # ,, label = 'Relative mean square error')
                    plt.plot(range(0, args.max_step + 1), np.abs(Listloss_std), label='Loss_std')
                    axes = plt.gca()
                    axes.set_yscale('log')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')  # plt.xlabel('Time (Lyapunov time)')
                    plt.legend()
                    ax = plt.subplot(2, 1, 2)
                    axes = plt.gca()
                    axes.set_yscale('log')
                    plt.plot(
                        range(0, args.max_step + 1),
                        np.abs(ListDistanceCheck_Eucli),
                        label='Euclidean distance',
                    )  # ,, label = 'Relative mean square error')
                    plt.plot(
                        range(0, args.max_step + 1), np.abs(ListDistanceCheck), label='KL-divergence'
                    )  # ,, label = 'Relative mean square error')
                    plt.ylabel('Distance')
                    plt.xlabel('Epoch')  # plt.xlabel('Time (Lyapunov time)')
                    fig = plt.gcf()
                    plt.legend()
                    fig.set_size_inches(8, 8)
                    plt.tight_layout()
                    # , 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)
                    plt.savefig(
                        '{}_img/TimeStep{}lambda{}.jpg'.format(args.out_filename, Tstep, lambda_tilt), dpi=300
                    )
                    plt.close()

            ListDynPartiFuncLog2_3 = np.array(ListDynPartiFuncLog2_3)
            ListDynPartiFuncLog2_3 = ListDynPartiFuncLog2_3[
                int(1 / args.delta_t) - 1 : -1 : int(1 / args.delta_t)
            ]

            # plt.figure(num=None,  dpi=400, edgecolor='k')
            # fig, axes = plt.subplots(1,1)
            # plt.plot(np.arange(len(ListDynPartiFuncLog2_3))+1, ListDynPartiFuncLog2_3/args.size, 'kx',markersize=4,label='VAN')#,, label = 'Relative mean square error')
            # if args.size<=10:
            #     ListDynPartiFuncLog3=np.array(ListDynPartiFuncLog3)
            #     ListDynPartiFuncLog3=ListDynPartiFuncLog3[int(1/args.delta_t)-1:-1:int(1/args.delta_t)]
            #     #np.savez('{}_img/DataPartition0'.format(args.out_filename),ListDynPartiFuncLog2,ListDynPartiFuncLog3,TP_ExactRecord)
            #     plt.plot(np.arange(len(ListDynPartiFuncLog3))+1, ListDynPartiFuncLog3/args.size, 'b',linewidth=0.8,label='Numerical exact')#,, label = 'Relative mean square error')
            # plt.xlim((1, len(ListDynPartiFuncLog2_3)))
            # axes = plt.gca()
            # axes.set_xscale('log')
            # axes.set_yscale('log')
            # plt.ylabel('-Log Z/L')
            # plt.xlabel('Time')#plt.xlabel('Time (Lyapunov time)')
            # plt.legend()
            # fig.set_size_inches(9, 6)
            # plt.savefig('{}_img/Partitionlambda{}.jpg'.format(args.out_filename,lambda_tilt), dpi=300)#, 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)

            Loss1 = Loss1[int(1 / args.delta_t) - 1 : -1 : int(1 / args.delta_t)]
            Loss1_2 = Loss1_2[int(1 / args.delta_t) - 1 : -1 : int(1 / args.delta_t)]
            Loss2 = Loss2[int(1 / args.delta_t) - 1 : -1 : int(1 / args.delta_t)]
            SummaryListDynPartiFuncLog2.append(ListDynPartiFuncLog2_3)
            SummaryListDynPartiFuncLog3.append(ListDynPartiFuncLog3)
            SummaryLoss1.append(Loss1)
            SummaryLoss2.append(Loss2)
            SummaryLoss1_2.append(Loss1_2)
            SummarySampleSum.append(SampleSum)
            print(np.array(SummarySampleSum).shape)
            argsSave = [args.Tstep, args.delta_t, args.L, args.dlambda, args.dlambdaL, args.dlambdaR, args.c]
            np.savez(
                '{}_img/DataL{}c{}s{}'.format(args.out_filename, args.L, args.c, args.dlambdaL),
                np.array(SummaryListDynPartiFuncLog2),
                argsSave,
                np.array(SummaryListDynPartiFuncLog3),
                np.array(SummaryLoss1),
                np.array(SummaryLoss2),
                np.array(SampleSum),
                np.array(SummarySampleSum),
                np.array(SummaryLoss1_2),
            )

            # if args.size<=10:
            #     DeltaLogZ2_3=np.abs((ListDynPartiFuncLog2_3-ListDynPartiFuncLog3)/ListDynPartiFuncLog3)
            #     #print(ListDynPartiFuncLog3)
            #     plt.figure(num=None,  dpi=400, edgecolor='k')
            #     fig, axes = plt.subplots(1,1)
            #     #fig.tight_layout()
            #     #plt.plot(np.arange(len(ListDynPartiFuncLog2))+1, DeltaLogZ2, 'rx',markersize=8,label='VAN')#,, label = 'Relative mean square error')
            #     #plt.plot(np.arange(len(ListDynPartiFuncLog2))+1, DeltaLogZ2_2, 'gx',markersize=8,label='VAN2')#,, label = 'Relative mean square error')
            #     plt.plot(np.arange(len(ListDynPartiFuncLog2_3))+1, DeltaLogZ2_3, 'kx',markersize=8,label='VAN')#,, label = 'Relative mean square error')
            #     plt.xlim((1, len(ListDynPartiFuncLog2_3)))
            #     Min=np.min([np.min(DeltaLogZ2_3)])
            #     Max=np.max([np.max(DeltaLogZ2_3)])
            #     plt.ylim((Min,Max))
            #     axes = plt.gca()
            #     axes.set_xscale('log')
            #     axes.set_yscale('log')
            #     plt.ylabel('Delta Log Z')
            #     plt.xlabel('Time')#plt.xlabel('Time (Lyapunov time)')
            #     plt.legend()
            #     fig.set_size_inches(9, 6)
            #     plt.savefig('{}_img/PartitionError{}.jpg'.format(args.out_filename,lambda_tilt), dpi=400)#, 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)

    end_time2 = time.time()
    print('Time ', (end_time2 - start_time2) / 60)
    print('Time ', (end_time2 - start_time2) / 3600)
    SummaryListDynPartiFuncLog2 = np.array(SummaryListDynPartiFuncLog2)
    SummaryListDynPartiFuncLog3 = np.array(SummaryListDynPartiFuncLog3)
    SummaryLoss1 = np.array(SummaryLoss1)
    SummaryLoss2 = np.array(SummaryLoss2)
    SampleSum = np.array(SampleSum)
    SummarySampleSum = np.array(SummarySampleSum)
    # np.savez('{}_img/DataL{}c{}s{}'.format(args.out_filename,args.L,args.c,args.dlambdaL),SummaryListDynPartiFuncLog2,argsSave,SummaryListDynPartiFuncLog3,SummaryLoss1,SummaryLoss2,SampleSum)
    np.savez(
        '{}_img/DataL{}c{}s{}'.format(args.out_filename, args.L, args.c, args.dlambdaL),
        SummaryListDynPartiFuncLog2,
        argsSave,
        SummaryListDynPartiFuncLog3,
        SummaryLoss1,
        SummaryLoss2,
        SampleSum,
        SummarySampleSum,
        np.array(SummaryLoss1_2),
    )

    # plt.figure(num=None,  dpi=400, edgecolor='k')
    # fig, axes = plt.subplots(1,1)
    # fig.tight_layout()
    # print(SummaryLoss1/args.size)
    # plt.plot(lambda_tilt_Range, SummaryLoss1[:,-1]/args.size, 'bo',markersize=8,label='L '+str(args.L))
    # plt.xlim((1e-4,0.8))
    # plt.ylim((1e-5,1e-1))
    # axes.set_yscale('log')
    # axes.set_xscale('log')
    # plt.xlabel('lambda')
    # plt.ylabel('-theta/L')#plt.xlabel('Time (Lyapunov time)')
    # plt.legend()
    # fig.set_size_inches(5, 5)
    # plt.savefig('{}_img/theta.jpg'.format(args.out_filename), dpi=400)#, 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)


if __name__ == '__main__':
    Test()
