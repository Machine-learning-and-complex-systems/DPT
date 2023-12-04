import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')
group.add_argument('--ham', type=str, default='fm', choices=['afm', 'fm'], help='Hamiltonian model')

group.add_argument(
    '--Model',
    type=str,
    default='No',
    choices=[
        'No',
        '1DFA',
        '1DEast',
        '2DFA',
        '3DFA',
        '3DEast',
        '3DNoEast',
        '3DSouthEast',
        '3DSouthEastBack',
        '2DEast',
        '2DNoEast',
        '2DSouthEast',
        '2DNorthWest',
        'Voter',
    ],
    help='Kinentically constrained model: 0 is no neighbor effect, 1 is FA Model, 2 is East Model',
)
group.add_argument(
    '--Tstep', type=int, default=1, help='Time step of iterating the dynamical equation P_tnew=T*P_t'
)
group.add_argument('--Number2', type=int, default=2, help='Number of added to batch')
group.add_argument('--ISNumber1', type=int, default=1, help='Number of IS')
group.add_argument('--lambda_tilt', type=float, default=0, help='tilt parameter for the generating function')
group.add_argument('--binomialP', type=float, default=0.5, help='binomialP of IS')
group.add_argument(
    '--delta_t', type=float, default=0.05, help='Time step length of iterating the dynamical equation'
)
group.add_argument('--c', type=float, default=0.5, help='Flip-up probability')
group.add_argument('--dlambda', type=float, default=0.2, help='steplength for the tilt parameter')
group.add_argument(
    '--dlambdaL', type=float, default=-2, help='10^L: Left bounary of scan for the tilt parameter'
)
group.add_argument(
    '--dlambdaR', type=float, default=0, help='10^R: Right bounary of scan for the tilt parameter'
)
group.add_argument('--Dim', type=int, default=2, help='Dimension of the lattice system')

group.add_argument(
    '--lossType',
    type=str,
    default='kl',
    choices=['l2', 'kl', 'klreweight', 'he', 'ss', 'ss1', 'ss2', 'ss3'],
    help='Loss functions: l2, KL-divergence, and Hellinger',
)

group.add_argument('--lattice', type=str, default='sqr', choices=['sqr', 'tri'], help='lattice shape')
group.add_argument(
    '--boundary', type=str, default='periodic', choices=['open', 'periodic'], help='boundary condition'
)
group.add_argument('--L', type=int, default=3, help='number of sites on each edge of the lattice')

group.add_argument('--lr_schedule_type', type=int, default=1, help='lr rate schedulers')

group.add_argument('--beta', type=float, default=1, help='beta = 1 / k_B T')

group = parser.add_argument_group('network parameters')
group.add_argument(
    '--net',
    type=str,
    default='pixelcnn',
    choices=['made', 'made3d', 'rnn', 'lstm', 'rnn2', 'rnn3', 'pixelcnn', 'stackedpixelcnn', 'bernoulli'],
    help='network type',
)
group.add_argument('--net_depth', type=int, default=3, help='network depth')
group.add_argument('--net_width', type=int, default=64, help='network width')
group.add_argument('--half_kernel_size', type=int, default=1, help='(kernel_size - 1) // 2')
group.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], help='dtype')
group.add_argument('--bias', action='store_true', help='use bias')
group.add_argument('--addBatch', action='store_true', help='addBatch')
group.add_argument('--IS', action='store_true', help='IS')
group.add_argument('--SwitchOffIS', action='store_true', help='SwitchOffIS')

group.add_argument('--reverse', action='store_true', help='with reverse conditional probability')

# group.add_argument('--BC', action='store_true', help='use left up boundary')
group.add_argument(
    '--BC', type=int, default=0, help='use up boundary: 1 is onlyl left boundary, 2 is all boundary'
)

group.add_argument('--Doob', action='store_true', help='Doob')
group.add_argument('--z2', action='store_true', help='use Z2 symmetry in sample and loss')
group.add_argument('--res_block', action='store_true', help='use res block')
group.add_argument(
    '--x_hat_clip', type=float, default=0, help='value to clip x_hat around 0 and 1, 0 for disabled'
)
group.add_argument('--final_conv', action='store_true', help='add an additional conv layer before sigmoid')
group.add_argument(
    '--epsilon', type=float, default=0, help='small number to avoid 0 in division and log'  # default=1e-39,
)

group = parser.add_argument_group('optimizer parameters')
group.add_argument('--seed', type=int, default=0, help='random seed, 0 for randomized')
group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['sgd', 'sgdm', 'rmsprop', 'adam', 'adam0.5'],
    help='optimizer',
)
group.add_argument('--batch_size', type=int, default=10**3, help='number of samples')
group.add_argument('--loadTime', type=int, default=1000, help='loadTime')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')
group.add_argument('--lrLoad', type=float, default=1e-3, help='learning rate for load VAN')
group.add_argument('--Percent', type=float, default=0.2, help='Percent of keeping the last epochs')
group.add_argument('--max_step', type=int, default=10**3, help='maximum number of steps')
group.add_argument('--max_stepAll', type=int, default=10**4, help='maximum number of steps')
group.add_argument(
    '--max_stepLoad', type=int, default=200, help='maximum number of steps of later time step by loading VAN'
)
group.add_argument('--max_stepLater', type=int, default=50, help='maximum number of steps of later time step')
group.add_argument(
    '--max_stepLater0',
    type=int,
    default=50,
    help='maximum number of steps of later time step before certain time point',
)
group.add_argument(
    '--max_stepLater2',
    type=int,
    default=50,
    help='maximum number of steps of later time step after certain time point',
)
group.add_argument('--max_stepT1', type=int, default=1, help='Time point 1 to change max_stepLater')
group.add_argument('--max_stepT2', type=int, default=5000, help='Time point 2 to change max_stepLater')
group.add_argument('--lr_schedule', action='store_true', help='use learning rate scheduling')
group.add_argument('--loadVAN', action='store_true', help='load VAN at later time points')
group.add_argument('--compareVAN', action='store_true', help='compareVAN')
group.add_argument('--neighborS', action='store_true', help='neighborS')
group.add_argument('--UseNewVAN', action='store_true', help='UseNewVAN or not')
group.add_argument('--negativeS', action='store_true', help='for negative s values, 10-exponent of nu=1-e^s')
group.add_argument('--Hermitian', action='store_true', help='Hermitian operator: necessary for c\neq0')
group.add_argument(
    '--beta_anneal', type=float, default=0, help='speed to change beta from 0 to final value, 0 for disabled'
)
group.add_argument('--clip_grad', type=float, default=0, help='global norm to clip gradients, 0 for disabled')

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--no_stdout', action='store_true', help='do not print log to stdout, for better performance'
)
group.add_argument('--clear_checkpoint', action='store_true', help='clear checkpoint')
group.add_argument('--print_step', type=int, default=1, help='number of steps to print log, 0 for disabled')
group.add_argument(
    '--save_step', type=int, default=100, help='number of steps to save network weights, 0 for disabled'
)
group.add_argument(
    '--visual_step', type=int, default=100, help='number of steps to visualize samples, 0 for disabled'
)
group.add_argument('--save_sample', action='store_true', help='save samples on print_step')
group.add_argument(
    '--print_sample',
    type=int,
    default=1,
    help='number of samples to print to log on visual_step, 0 for disabled',
)
group.add_argument(
    '--print_grad', action='store_true', help='print summary of gradients for each parameter on visual_step'
)
group.add_argument('--cuda', type=int, default=-1, help='ID of GPU to use, -1 for disabled')
group.add_argument(
    '--out_infix', type=str, default='', help='infix in output filename to distinguish repeated runs'
)
group.add_argument(
    '-o', '--out_dir', type=str, default='out', help='directory prefix for output, empty for disabled'
)

args = parser.parse_args()
