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
from Transition3D import (
    SCGF,
    AllTransitionState1D,
    OptimizeFunction,
    Optimizer,
    TransitionState,
    gen_all_binary_vectors,
)
from stacked_pixelcnnFA import StackedPixelCNN
from pixelcnn import PixelCNN
from mdtensorizedrnn import MDTensorizedRNN
from mdrnn import RNN2D
from made3d import MADE3D
from made1D import MADE1D
from made import MADE
from lstm2D import LSTM2D
from gru2D import GRU2D
from bernoulli import BernoulliMixture
from args import args
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


plt.rc('font', size=16)


def Test():
    # #Initialize parameters: see args.py for a the help information on the parameters if needed
    # #System parameters:
    args.Model = (
        '3DSouthEastBack'  # type of models: 'No','FA', 'East', see args.py for a the help information
    )
    args.L = 2  # Lattice size: 1D
    args.Tstep = 11  # Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
    args.delta_t = 0.1  # Time step length
    args.dlambda = (
        0.1  # The steplength from the left to the rigth boundary value of the count field  (lambda=s)
    )
    args.dlambdaL = -1.6  # left boundary value of the count field  (lambda=s)
    args.dlambdaR = -1.55  # Rigth boundary value of the count field  (lambda=s)
    args.c = 0.5  # 0.2#0.5 #Flip-up probability
    args.negativeS = True  # For the negative counting field (s<0) or not (s>0)
    # #Neural-network hyperparameters:
    args.net = 'made3d'  # 'rnn'#'lstm'#'rnn'Type of neural network in the VAN
    args.max_stepAll = 500  # 0 #The epoch for the 1st  time steps
    args.max_stepLater = 50  # 00 #The epoch at time steps except for the 1st step
    args.print_step = 1  # 0   # The time step size of print and save results
    args.net_depth = 3  # 3#3  # Depth of the neural network
    args.net_width = 2  # 128 # Width of the neural network
    args.batch_size = 100  # 1000 #batch size

    ###################################################
    # #Optional initial parameters
    # args.IS=True #Importance sampling
    # args.ISNumber1=2 #The number of samples for importance sampling
    # args.SwitchOffIS=True #Switch-off importance sampling
    # # args.half_kernel_size=2 #kernel size for pixelcnn
    # # args.loadVAN=False# Load the saved VAN at some time steps
    # args.loadTime=0# The loaded time steps
    # args.lrLoad=0.001# The loaded lr
    # args.max_stepLoad=100 # The epoch after the load
    # args.compareVAN=False#True # Compare the loaded VAN and new VAN, or not
    # args.Hermitian=False # Do Hermitian transform or not
    # args.Doob=False #Use Doob operator
    # args.BC=0 #Other boundary conditions
    # args.reverse=False #Use reverse order in RNN

    ###################################################
    # #Default parameters
    args.max_step = args.max_stepAll  # args.max_step=max_step
    args.lossType = 'kl'  # 'ss'# #type of loss
    args.clip_grad = 1  # clip gradient
    # args.lr_schedule=1#1, 2
    args.bias = True  # With bias or not in the neural network
    args.size = args.L * args.L * args.L  # the number of spin: 1D, doesnt' count the boundary spins
    args.binomialP = round(2 / args.size, 3)
    args.epsilon = 1e-300  # 0#1e-8/(2**args.size) # avoid 0 value in log below
    args.free_energy_batch_size = args.batch_size  # 1000
    # args.Hermitian=False # Do Hermitian transform or not
    lambda_tilt_Range = 10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda))  # plot heatmap
    if args.negativeS:
        lambda_tilt_Range = np.log(
            1 - 10 ** (np.arange(args.dlambdaL, args.dlambdaR, args.dlambda))
        )  # plot heatmap
    start_time2 = time.time()

    # start_time = time.time()
    init_out_dir()
    SummaryListDynPartiFuncLog2 = []
    SummaryListDynPartiFuncLog2_2 = []
    SummaryListDynPartiFuncLog3 = []
    net_new = []
    SummaryLoss1 = []
    SummaryLoss1_2 = []
    SummaryLoss2 = []
    SummarySampleSum = []

    # This is to get numerical exact result for small systems:
    if args.size <= 8:
        # Configuration of spins: from left to right, up to down for the lattice
        Sample1D = gen_all_binary_vectors(args.size - 1)
        Sample1DExtend = torch.cat((torch.ones(Sample1D.shape[0], 1), Sample1D), 1)
        ones = torch.sum((Sample1D == 1.0), 1)
        # BatchSize   *scipy.special.binom(args.size, ones) #No binomal coefficient for each element
        P_ss = args.c**ones * (1 - args.c) ** (args.size - 1 - ones)
        PInverse12 = 1 / torch.sqrt(P_ss)
        # All possible 1-spin flipped configurations to the sampled state: NeighborSize X BatchSize X SystemSize
        SampleNeighbor1D1 = Sample1D.repeat(args.size - 1, 1, 1).permute(1, 0, 2)
        SampleNeighbor1D2 = (SampleNeighbor1D1 - 1).abs()
        Mask = torch.eye(args.size - 1).expand(Sample1D.shape[0], args.size - 1, args.size - 1)
        SampleNeighbor1D = (SampleNeighbor1D1 * (1 - Mask) + SampleNeighbor1D2 * Mask).permute(
            1, 0, 2
        )  # BatchSize X SystemSize X SystemSize
        ones_other = (SampleNeighbor1D == 1.0).sum(dim=2)  # SystemSize X BatchSize
        # Correct: BatchSize X NeighborSize (flipped spin state)
        P_ss_other = (args.c**ones_other * (1 - args.c) ** (args.size - 1 - ones_other)).t()

        Dhamming = (torch.cdist(Sample1D, Sample1D, p=0) == 1).nonzero(
            as_tuple=False
        )  # hamming distance by p=0 ##.nonzero(as_tuple=True)
        IdConnec = torch.zeros(2 ** (args.size - 1), args.size - 1)
        for i in range(2 ** (args.size - 1)):
            IdConnecTemp = Dhamming[
                Dhamming[:, 0] == i, 1
            ]  # The index of "transmissible" state to the i-th state
            IdFliped = torch.nonzero([Sample1D[i] - Sample1D[IdConnecTemp] != 0][0].int())[
                :, 1
            ]  # The position of the flipped spin
            IdConnec[i, :] = Dhamming[Dhamming[:, 0] == i, 1][
                np.argsort(IdFliped.numpy())
            ]  # Correct way to ordering
        Sample2D = Sample1DExtend.view(
            -1, args.L, args.L, args.L
        )  # Correct: sample has size batchsize X L X L
        # Correct: The connected states flip into this sampled state
        Win = (Sample1D - 1).abs() * (1 - args.c) + Sample1D * args.c
        Col1 = torch.cat(
            (torch.zeros(Sample2D.shape[0], args.L, 1, args.L), Sample2D[:, :, :-1, :]), 2
        )  # down sites
        Col2 = torch.cat((Sample2D[:, :, 1:, :], torch.zeros(Sample2D.shape[0], args.L, 1, args.L)), 2)
        Row1 = torch.cat((torch.zeros(Sample2D.shape[0], 1, args.L, args.L), Sample2D[:, :-1, :, :]), 1)
        Row2 = torch.cat((Sample2D[:, 1:, :, :], torch.zeros(Sample2D.shape[0], 1, args.L, args.L)), 1)
        Third1 = torch.cat((torch.zeros(Sample2D.shape[0], args.L, args.L, 1), Sample2D[:, :, :, :-1]), 3)
        Third2 = torch.cat((Sample2D[:, :, :, 1:], torch.zeros(Sample2D.shape[0], args.L, args.L, 1)), 3)
        if args.BC == 1:
            Col1 = torch.cat((torch.ones(Sample2D.shape[0], args.L, 1), Sample2D[:, :, :-1]), 2)  # down sites
        if args.BC == 2:
            Col1 = torch.cat((torch.ones(Sample2D.shape[0], args.L, 1), Sample2D[:, :, :-1]), 2)  # down sites
            Col2 = torch.cat((Sample2D[:, :, 1:], torch.ones(Sample2D.shape[0], args.L, 1)), 2)
            Row1 = torch.cat((torch.ones(Sample2D.shape[0], 1, args.L), Sample2D[:, :-1, :]), 1)
            Row2 = torch.cat((Sample2D[:, 1:, :], torch.ones(Sample2D.shape[0], 1, args.L)), 1)
        if args.BC == 3:
            Col1 = torch.cat(
                (Sample2D[:, :, -1].view(Sample2D.shape[0], args.L, 1), Sample2D[:, :, :-1]), 2
            )  # down sites
            Col2 = torch.cat((Sample2D[:, :, 1:], Sample2D[:, :, 0].view(Sample2D.shape[0], args.L, 1)), 2)
            Row1 = torch.cat((Sample2D[:, -1, :].view(Sample2D.shape[0], 1, args.L), Sample2D[:, :-1, :]), 1)
            Row2 = torch.cat((Sample2D[:, 1:, :], Sample2D[:, 0, :].view(Sample2D.shape[0], 1, args.L)), 1)

        if args.Model == '2DFA':
            # Correct: torch.cat((torch.ones(Sample2D.shape[0],args.L,1),Sample2D[:,:-1,:]) ,2)+torch.cat((Sample1D[:,1:],torch.ones(Sample1D.shape[0],1)) ,1)
            fNeighbor = (Col1 + Col2 + Row1 + Row2).view(-1, args.size)
        if args.Model == '3DFA':
            # torch.cat((torch.ones(Sample2D.shape[0],args.L,1),Sample2D[:,:-1,:]) ,2)+torch.cat((Sample1D[:,1:],torch.ones(Sample1D.shape[0],1)) ,1)
            fNeighbor = (Col1 + Col2 + Row1 + Row2 + Third1 + Third2).view(-1, args.size)
        if args.Model == '3DEast':
            # torch.cat((torch.ones(Sample2D.shape[0],args.L,1),Sample2D[:,:-1,:]) ,2)+torch.cat((Sample1D[:,1:],torch.ones(Sample1D.shape[0],1)) ,1)
            fNeighbor = (Col1).view(-1, args.size)
        if args.Model == '3DSouthEast':
            fNeighbor = (Col1 + Row1).view(-1, args.size)
        if args.Model == '3DSouthEastBack':
            fNeighbor = (Col1 + Row1 + Third1).view(-1, args.size)
        # To calculate R: use DB condition, the current state flip out to connected state
        WinTemp = (1 - Win) * fNeighbor[:, 1:]
        # Correct: the sum of transiiton out. BatchSize: The escape-probability for each sampled state to all connected states#torch.sum((Sample1D-1).abs()*args.c+Sample1D*(1-args.c),1)
        R = torch.sum(WinTemp, 1)
        # Correct: BatchSize X NeighborSize: The in-probability for each previous-step state flipped into the sampled state
        Win = Win * fNeighbor[:, 1:]
        dict2 = {
            'Sample1D': Sample1D,
            'SampleNeighbor1D': SampleNeighbor1D,
            'IdConnec': IdConnec,
            'P_ss': P_ss,
            'P_ss_other': P_ss_other,
            'Win': Win,
            'WinTemp': WinTemp,
            'R': R,
            'PInverse12': PInverse12,
        }

    count = -1
    for lambda_tilt in lambda_tilt_Range:
        count += 1
        args.lambda_tilt = lambda_tilt
        args.max_step = args.max_stepAll  # args.max_step=max_step
        # Initialize net and optimizer
        if args.net == 'made':
            net = MADE(**vars(args))  # net2 = MADE1D(**vars(args))
        if args.net == 'made3d':
            net = MADE3D(**vars(args))  # net2 = MADE1D(**vars(args))
            net1 = MADE3D(**vars(args))
        elif args.net == 'rnn':
            net = GRU2D(**vars(args))
        elif args.net == 'lstm':
            net = LSTM2D(**vars(args))
        elif args.net == 'rnn2':
            net = RNN2D(**vars(args))
        elif args.net == 'rnn3':
            net = MDTensorizedRNN(**vars(args))
        elif args.net == 'pixelcnn':
            net = PixelCNN(**vars(args))
        elif args.net == 'stackedpixelcnn':
            net = StackedPixelCNN(**vars(args))
        elif args.net == 'bernoulli':
            net = BernoulliMixture(**vars(args))
        else:
            raise ValueError('Unknown net: {}'.format(args.net))
        net.to(args.device)

        TP_Exact = []
        ListDynPartiFuncLog3 = []
        ListDynPartiFuncLog3A = []
        DynPartiFuncLog3 = 0
        ListDynPartiFuncLog2_3 = []
        ListDynPartiFuncLog2_3_2 = []
        DynPartiFuncLog2_3 = 0
        DynPartiFuncLog2_3_2 = 0
        Loss1 = []
        Loss1_2 = []
        Loss2 = []
        TP_ExactRecord = []
        SampleSum = []
        if args.addBatch:
            label = torch.arange(0, args.size).long()  # torch.tensor([1:1:2])  # 2显示的是索引
            num_class = args.size
            label2one_hot = torch.nn.functional.one_hot(label, num_classes=num_class)
            if args.net == 'rnn':
                addBatch = label2one_hot.reshape(-1, args.L, args.L) * 2 - 1
            else:
                addBatch = label2one_hot.reshape(-1, 1, args.L, args.L) * 2 - 1
        else:
            addBatch = []

        # Load NN starting from certain time point:
        startT = 0
        if args.loadVAN:
            PATH = args.out_filename + str(round(args.lambda_tilt, 3)) + 'Tstep' + str(args.loadTime)
            startT = args.loadTime
            print(PATH)
            if args.cuda == -1:
                state = torch.load(PATH, map_location=torch.device('cpu'))  # CPU
            else:
                state = torch.load(PATH)  # GPU
            if not args.UseNewVAN:
                net1 = GRU2D(**vars(args))
                net1.load_state_dict(state['net'])  # new save format
                net1.to(args.device)
                if args.compareVAN:
                    optimizer1, params1, nparams1 = Optimizer(
                        net1, args
                    )  # From steady-state VAN: not evolve itself
                    optimizer1.load_state_dict(state['optimizer'])  # load saved optimizer1
                print('Use saved VAN')
                if args.lossType == 'ss':  # or args.lossType=='kl':
                    net = copy.deepcopy(net1)
                    print('Use saved VAN for ss')
            net_new = copy.deepcopy(net)  # net
            net_new.to(args.device)
            net_new.requires_grad = False
            if args.compareVAN:
                args.out_filename = (
                    args.out_filename
                    + 'CompareVAN'
                    + str(args.loadTime)
                    + 'lr'
                    + str(args.lrLoad)
                    + 'epoch'
                    + str(args.max_stepLoad)
                )
            else:
                args.out_filename = (
                    args.out_filename
                    + 'loadVAN'
                    + str(args.loadTime)
                    + 'lr'
                    + str(args.lrLoad)
                    + 'epoch'
                    + str(args.max_stepLoad)
                )
            ensure_dir(args.out_filename + '_img/')

        optimizer, params, nparams = Optimizer(net, args)

        if args.Doob == 1:
            data = np.load(
                '{}_img/Data3DSSL{}c{}s{}'.format(args.out_filename, args.L, args.c, args.dlambdaL) + '.npz'
            )
            print(list(data))
            print(data['arr_3'].shape)
            args.thetaLoad = -data['arr_3'][count, -1]  # loss=-theta

        for Tstep in range(startT, args.Tstep):  # Time step of the dynamical equation
            # This is to get numerical exact result for small systems
            if args.size <= 8:  # compare with numerical exact result
                if Tstep == 0:
                    TP_Exact = dict2['P_ss'].to(
                        args.device
                    )  # [1:]#AllTransitionState1D(args,dict2,Tstep,TP_Exact)
                TP_Exact = AllTransitionState1D(args, dict2, Tstep, TP_Exact)
                TP_ExactRecord.append(TP_Exact.detach().cpu().numpy())

            args.ISNumber11 = 0
            if Tstep >= 1 and not args.loadVAN:
                if Tstep < args.max_stepT1:
                    args.max_step = args.max_stepLater0  # default 50 epoch
                elif Tstep >= args.max_stepT1 and Tstep < args.max_stepT2:
                    args.max_step = args.max_stepLater
                    args.ISNumber11 = args.ISNumber1
                    print(args.ISNumber11)
                else:
                    args.max_step = args.max_stepLater2  # default 50 epoch
            if args.loadVAN:
                if Tstep == startT:
                    args.max_step = args.max_stepAll
                else:
                    args.max_step = args.max_stepLoad

            scheduler = []
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

            # Train VAN: if args.loadVAN, use loaded VAN
            if not args.compareVAN:
                (
                    net,
                    optimizer,
                    SampleT,
                    free_energy_mean3Temp,
                    free_energy_mean3Temp2,
                    Loss1Temp,
                    Loss1_2Temp,
                    ListDistanceCheck_Eucli,
                    DistanceCheck_Eucli,
                    ListDistanceCheck,
                    Listloss_mean,
                    Listloss_std,
                    Listloss_mean22,
                    Listloss_std22,
                ) = OptimizeFunction(
                    addBatch, net, params, optimizer, scheduler, net_new, args, lambda_tilt, Tstep
                )

            # Compare VAN: if args.compareVAN, compare Null VAN and saved VAN at steady state for phase transitoin
            if args.compareVAN:
                print('Use two VANs to compare')
                (
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
                ) = OptimizeFunction(net, params, optimizer, scheduler, net_new, args, lambda_tilt, Tstep)

                net2 = copy.deepcopy(net1)
                [optimizer2, params2, nparams2] = [optimizer1, params1, nparams1]
                (
                    net2,
                    optimizer2,
                    SampleT2,
                    free_energy_mean3Temp2,
                    Loss1Temp2,
                    Loss1_2Temp2,
                    ListDistanceCheck_Eucli2,
                    DistanceCheck_Eucli2,
                    ListDistanceCheck2,
                    Listloss_mean2,
                    Listloss_std2,
                ) = OptimizeFunction(net2, params2, optimizer2, scheduler, net_new, args, lambda_tilt, Tstep)
                LossCompare1 = np.array(
                    torch.mean(torch.tensor(Loss1Temp, dtype=torch.float64).to(args.device)).detach().cpu()
                )
                LossCompare2 = np.array(
                    torch.mean(torch.tensor(Loss1Temp2, dtype=torch.float64).to(args.device)).detach().cpu()
                )
                if LossCompare1 > LossCompare2:
                    print('Steady state VAN is used')
                    net = copy.deepcopy(net2)
                    [
                        optimizer,
                        params,
                        SampleT,
                        free_energy_mean3Temp,
                        Loss1Temp,
                        Loss1_2Temp,
                        ListDistanceCheck_Eucli,
                        DistanceCheck_Eucli,
                        ListDistanceCheck,
                        Listloss_mean,
                        Listloss_std,
                    ] = [
                        optimizer2,
                        params2,
                        SampleT2,
                        free_energy_mean3Temp2,
                        Loss1Temp2,
                        Loss1_2Temp2,
                        ListDistanceCheck_Eucli2,
                        DistanceCheck_Eucli2,
                        ListDistanceCheck2,
                        Listloss_mean2,
                        Listloss_std2,
                    ]

            with torch.no_grad():
                free_energy_mean3 = torch.mean(
                    torch.tensor(free_energy_mean3Temp, dtype=torch.float64).to(args.device)
                )  # loss.mean() #/ args.beta / args.size
                free_energy_mean3_2 = torch.mean(
                    torch.tensor(free_energy_mean3Temp2, dtype=torch.float64).to(args.device)
                )  # loss.mean() #/ args.beta / args.size
                net_new = copy.deepcopy(net)  # net
                net_new.requires_grad = False
                net_new1 = copy.deepcopy(net1)  # net
                net_new1.requires_grad = False

                if (Tstep == 0 or Tstep == 100 or Tstep == 500 or Tstep == 1000) and not args.loadVAN:
                    PATH = args.out_filename + str(round(args.lambda_tilt, 3)) + 'Tstep' + str(Tstep)
                    # torch.save(net, PATH)
                    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, PATH)

                DynPartiFuncLog2_3 += free_energy_mean3.detach().cpu().numpy()
                ListDynPartiFuncLog2_3.append(DynPartiFuncLog2_3)
                DynPartiFuncLog2_3_2 += free_energy_mean3_2.detach().cpu().numpy()
                ListDynPartiFuncLog2_3_2.append(DynPartiFuncLog2_3_2)
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
                SampleSum.append(np.mean(np.array(SampleT).reshape(-1, args.L, args.L), 0))
                if args.size <= 16:
                    free_energy_mean33 = -torch.log(torch.sum(TP_Exact))
                    # Each time it has summed all the sequential partition functions
                    DynPartiFuncLog3 += free_energy_mean33.detach().cpu().numpy()
                    TP_Exact = torch.abs(TP_Exact / TP_Exact.sum())  # renormalize at each step
                    ListDynPartiFuncLog3.append(DynPartiFuncLog3)

            # #Plot training loss
            if (
                Tstep <= 1
                or Tstep == 500
                or Tstep == 1000
                or Tstep == 2000
                or Tstep == 3000
                or Tstep == 4000
                or Tstep == 10000
            ):
                plt.figure(num=None, dpi=300, edgecolor='k')
                fig, axes = plt.subplots(3, 1)
                fig.tight_layout()
                ax = plt.subplot(3, 1, 1)
                # ,, label = 'Relative mean square error')
                plt.plot(range(0, args.max_step + 1), np.abs(Listloss_mean), label='Loss_mean')
                if args.IS:
                    # ,, label = 'Relative mean square error')
                    plt.plot(range(0, args.max_step + 1), np.abs(Listloss_mean22), label='Loss_mean with IS')
                axes = plt.gca()
                axes.set_yscale('log')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')  # plt.xlabel('Time (Lyapunov time)')
                plt.legend()
                ax = plt.subplot(3, 1, 2)
                # ,, label = 'Relative mean square error')
                plt.plot(range(0, args.max_step + 1), np.abs(Listloss_std), label='Loss_std')
                if args.IS:
                    # ,, label = 'Relative mean square error')
                    plt.plot(range(0, args.max_step + 1), np.abs(Listloss_std22), label='Loss_std with IS')
                axes = plt.gca()
                axes.set_yscale('log')
                plt.ylabel('Loss std')
                plt.xlabel('Epoch')  # plt.xlabel('Time (Lyapunov time)')
                plt.legend()
                ax = plt.subplot(3, 1, 3)
                axes = plt.gca()
                axes.set_yscale('log')
                if args.lossType == 'ss':
                    plt.plot(
                        range(0, args.max_step + 1), np.abs(ListDistanceCheck_Eucli), label='std up spin'
                    )  # ,, label = 'Relative mean square error')
                    plt.plot(
                        range(0, args.max_step + 1), np.abs(ListDistanceCheck), label='mean up spin'
                    )  # ,, label = 'Relative mean square error')
                else:
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
                fig.set_size_inches(8, 12)
                plt.tight_layout()
                plt.savefig(
                    '{}_img/TimeStep{}lambda{}.jpg'.format(args.out_filename, Tstep, lambda_tilt), dpi=300
                )  # , 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)
                plt.close()

            ListDynPartiFuncLog2_3A = np.array(ListDynPartiFuncLog2_3)
            ListDynPartiFuncLog2_3_2A = np.array(ListDynPartiFuncLog2_3_2)

            if args.size <= 8:
                ListDynPartiFuncLog3A = np.array(ListDynPartiFuncLog3)
                DeltaLogZ2_3 = np.abs(
                    (ListDynPartiFuncLog2_3A - ListDynPartiFuncLog3A) / ListDynPartiFuncLog3A
                )
                plt.figure(num=None, dpi=400, edgecolor='k')
                fig, axes = plt.subplots(1, 1)
                plt.plot(
                    np.arange(len(ListDynPartiFuncLog2_3A)) + 1, DeltaLogZ2_3, 'kx', markersize=8, label='VAN'
                )  # ,, label = 'Relative mean square error')
                plt.xlim((1, len(ListDynPartiFuncLog2_3A) + 1))
                Min = np.min([np.min(DeltaLogZ2_3)])
                Max = np.max([np.max(DeltaLogZ2_3)])
                plt.ylim((Min, Max + 1e-10))
                axes = plt.gca()
                axes.set_xscale('log')
                axes.set_yscale('log')
                plt.ylabel('Delta Log Z')
                plt.xlabel('Time')  # plt.xlabel('Time (Lyapunov time)')
                plt.legend()
                fig.set_size_inches(9, 6)
                plt.savefig(
                    '{}_img/PartitionError{}.jpg'.format(args.out_filename, lambda_tilt), dpi=400
                )  # , 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)
                plt.cla()
                plt.close("all")

            # plt.figure(num=None,  dpi=400, edgecolor='k')
            # fig, axes = plt.subplots(1,1)
            # plt.plot(np.arange(len(ListDynPartiFuncLog2_3A))+1, ListDynPartiFuncLog2_3A/args.size, 'kx',markersize=4,label='VAN')#,, label = 'Relative mean square error')
            # if args.IS:
            #     plt.plot(np.arange(len(ListDynPartiFuncLog2_3_2A))+1, ListDynPartiFuncLog2_3_2A/args.size, 'rx',markersize=4,label='VAN IS')#,, label = 'Relative mean square error')
            # if args.size<=8:
            #     plt.plot(np.arange(len(ListDynPartiFuncLog2_3A))+1, ListDynPartiFuncLog3A/args.size, 'b',linewidth=0.8,label='Numerical exact')#,, label = 'Relative mean square error')
            # plt.xlim((1, len(ListDynPartiFuncLog2_3A)+1))
            # axes = plt.gca()
            # axes.set_xscale('log')
            # axes.set_yscale('log')
            # plt.ylabel('-Log Z/L^2')
            # plt.xlabel('Time')#plt.xlabel('Time (Lyapunov time)')
            # plt.legend()
            # fig.set_size_inches(9, 6)
            # plt.savefig('{}_img/Partitionlambda{}.jpg'.format(args.out_filename,lambda_tilt), dpi=300)#, 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)
            # plt.cla()
            # plt.close("all")

        if args.IS:
            SummaryListDynPartiFuncLog2.append(ListDynPartiFuncLog2_3_2A)
            SummaryListDynPartiFuncLog2_2.append(ListDynPartiFuncLog2_3A)
        else:
            SummaryListDynPartiFuncLog2.append(ListDynPartiFuncLog2_3A)
        SummaryListDynPartiFuncLog3.append(ListDynPartiFuncLog3A)
        SummaryLoss1.append(Loss1)
        SummaryLoss1_2.append(Loss1_2)
        SummaryLoss2.append(Loss2)
        SummarySampleSum.append(SampleSum)
        print(np.array(SummarySampleSum).shape)
        argsSave = [args.Tstep, args.delta_t, args.L, args.dlambda, args.dlambdaL, args.dlambdaR, args.c]
        np.savez(
            '{}_img/Data{}L{}c{}s{}'.format(args.out_filename, args.Model, args.L, args.c, args.dlambdaL),
            np.array(SummaryListDynPartiFuncLog2),
            argsSave,
            np.array(SummaryListDynPartiFuncLog3),
            np.array(SummaryLoss1),
            np.array(SummaryLoss2),
            np.array(SampleSum),
            np.array(SummarySampleSum),
            np.array(SummaryLoss1_2),
            np.array(SummaryListDynPartiFuncLog2_2),
        )

    end_time2 = time.time()
    print('Time ', (end_time2 - start_time2) / 60)
    print('Time ', (end_time2 - start_time2) / 3600)
    SummaryListDynPartiFuncLog2 = np.array(SummaryListDynPartiFuncLog2)
    SummaryListDynPartiFuncLog3 = np.array(SummaryListDynPartiFuncLog3)
    SummaryListDynPartiFuncLog2_2 = np.array(SummaryListDynPartiFuncLog2_2)
    SummaryLoss1 = np.array(SummaryLoss1)
    SummaryLoss2 = np.array(SummaryLoss2)
    SampleSum = np.array(SampleSum)
    SummarySampleSum = np.array(SummarySampleSum)
    np.savez(
        '{}_img/Data{}L{}c{}s{}'.format(args.out_filename, args.Model, args.L, args.c, args.dlambdaL),
        SummaryListDynPartiFuncLog2,
        argsSave,
        SummaryListDynPartiFuncLog3,
        SummaryLoss1,
        SummaryLoss2,
        SampleSum,
        SummarySampleSum,
        np.array(SummaryLoss1_2),
        SummaryListDynPartiFuncLog2_2,
    )
    if args.lossType == 'ss' or args.lossType == 'ss1' or args.lossType == 'ss2' or args.lossType == 'ss3':
        np.savez(
            '{}_img/DataSS{}L{}c{}s{}'.format(args.out_filename, args.Model, args.L, args.c, args.dlambdaL),
            SummaryListDynPartiFuncLog2,
            argsSave,
            SummaryListDynPartiFuncLog3,
            SummaryLoss1,
            SummaryLoss2,
            SampleSum,
            SummarySampleSum,
            np.array(SummaryLoss1_2),
            SummaryListDynPartiFuncLog2_2,
        )

    # plt.figure(num=None,  dpi=400, edgecolor='k')
    # fig, axes = plt.subplots(1,1)
    # fig.tight_layout()
    # print(SummaryLoss1/args.size)
    # plt.plot(lambda_tilt_Range, SummaryLoss1[:,-1]/args.size, 'bo',markersize=8,label='L '+str(args.L))
    # plt.xlim((1e-4,0.8))
    # plt.ylim((1e-5,1e1))
    # axes.set_yscale('log')
    # axes.set_xscale('log')
    # plt.xlabel('lambda')
    # plt.ylabel('-theta/L^2')#plt.xlabel('Time (Lyapunov time)')
    # plt.legend()
    # fig.set_size_inches(5, 5)
    # plt.savefig('{}_img/theta.jpg'.format(args.out_filename), dpi=400)#, 'Loss_L%g_TimeStep%g.jpg'%(args.L,args.Tstep), dpi=300)


if __name__ == '__main__':
    Test()
