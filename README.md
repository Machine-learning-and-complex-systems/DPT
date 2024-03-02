(c) Lab of Machine Learning and Complex Systems (2023).
All rights reserved. 

A software package for the manuscript "Learning nonequilibrium statistical mechanics and dynamical phase transitions" [https://www.nature.com/articles/s41467-024-45172-8]

DPT stands for dynamical phase transitions. It tracks the time evolution of nonequilibrium statistical mechanics by training the variational autoregressive neural network, which generates the time evolution of the joint probability distribution and the dynamical partition function. This approach is applicable to uncover dynamical phase transitions, such as for the spin-flip activity. 


--------------------------------------------------------------------------------------------------------------------------------------------

System requirements: 
All simulations were done using Python.
We have used the package Pytorch. The code requires Python >= 3.6 and PyTorch >= 1.0.

--------------------------------------------------------------------------------------------------------------------------------------------

# Inputs

Please run Main2D.py on PC. The main files of 1D, 3D are Main1D.py, Main3D.py. The main file of the Voter model is MainVoter.py.

The system parameters and neural-network hyperparameters are in these scripts. 

## Note: 

(a) For hyperparameters such as dt, please use those in the Main scripts as a reference.

(b) After training, one can plot the result using `Data.npz` in the `out` folder.

(c) See args.py for a the help information on the parameters if needed.

(d) When running long time steps, please set args.print_step to be larger, such that the results are printed/saved less often and have smaller size.

--------------------------------------------------------------------------------------------------------------------------------------------

# Platforms

To implement the code after providing the `.py` file, there are two ways:

(a) PC users (Windows): you may use Spyder to run Main2D.py. You can properly adjust the hyperparameters in Main2D.py, including those listed below.

(b) Server: you can use a `.sh` to input the hyperparameters from the shell. Please refer to scripts `.sh` in the folder "Shell".
If you want run on a server, please comment out the section of initial parameters and use sh files in the folder `Shell` to input the parameters from the shell  (see below). 


```
    # #Initialize parameters
    # #System parameters:
    args.Model='2DSouthEast' # type of models: 'No','FA', 'East', see args.py for a the help information
    args.L=2# Lattice size: 1D  
    args.Tstep=11#Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
    args.delta_t=0.1#Time step length
    args.dlambda=0.1#The steplength from the left to the rigth boundary value of the count field  (lambda=s)
    args.dlambdaL=-1.6 #left boundary value of the count field  (lambda=s)
    args.dlambdaR=-1.55 #Rigth boundary value of the count field  (lambda=s)
    args.c=0.5#0.2#0.5 #Flip-up probability
    args.negativeS=True # For the negative counting field (s<0) or not (s>0)
    # #Neural-network hyperparameters:
    args.net ='rnn'#'rnn'#'lstm'#'rnn'Type of neural network in the VAN
    args.max_stepAll=500#0 #The epoch for the 1st  time steps 
    args.max_stepLater=50#00 #The epoch at time steps except for the 1st step
    args.print_step=1#0   # The time step size of print and save results
    args.net_depth=1#3#3  # Depth of the neural network
    args.net_width=16#128 # Width of the neural network
    args.batch_size=100#1000 #batch size  
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Examples

Examples of the methods are given in the main text. The representative examples include:  

(1) Voter model,

(2) 2D Fredrickson-Andersen model, 

(3) 2D South-East model, 

(4) 3D South-East-Front model, 

(5) 1D Fredrickson-Andersen model, 

(6) 1D East model.

--------------------------------------------------------------------------------------------------------------------------------------------

# Results

Jupiter notebooks for plotting the figures in the folder `PlotTemplate`. Genearlly, it needs to run the data from the main files first. Please generate the data files first and then change the file name to plot them. 
In this folder, there is a folder `plot_KL_divergence`, which provides a Jupiter notebook for plotting the KL-divergence or loss function over time points, by loading the data already saved. It is for quantifying the accuracy of the method, with the 2D SE model as a demonstrative example.

Scripts `.sh` in the folder "Shell' are commands to reproduce the results. Directly running these scripts several GPU hours. Expected run time for the examples are provided in the Supplmentary Information of the manuscript: All computations are performed with a single core GPU (~25% usage) of a Tesla-V100. In practice, one may run these commands with different hyperparameters in parallel on multiple GPUs.

To fully reproduce the result of phase transitions and emergence of the active phases, one needs to scan various values of the counting field. One also needs to run the simulation sufficiently long time to get the full time regime.

--------------------------------------------------------------------------------------------------------------------------------------------

If you have any questions or need help, feel free to contact us.

Contact: Ying Tang, jamestang23@gmail.com

