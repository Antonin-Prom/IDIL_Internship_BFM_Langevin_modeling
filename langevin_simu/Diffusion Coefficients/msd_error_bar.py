from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from class_diffusion_potential import *

simu1 = DiffusionSimulation(frequency = 26,torque  = 15 )

def expe_msd_error_bar():
    mean_msd, error_size = simu1.mean_msd_with_variance(50,5000,2)
    time_array = np.arange(0, len(mean_msd), 1)*simu1.dt_s 
    plt.plot(time_array, mean_msd)
    plt.xlabel('time [s]')
    plt.ylabel('mean square displacement [rad^2]')

    mean_msd= mean_msd[:-1]
    time_array = time_array[:-1]
    nb_error_bar = 10
    L = len(time_array)
    time_array_err = []
    mean_msd_err = []
    error_size_err = []
    for i in range(1,nb_error_bar):
        time_array_err.append(time_array[L*i//nb_error_bar])
        mean_msd_err.append(mean_msd[L*i//nb_error_bar])
        error_size_err.append(error_size[L*i//nb_error_bar])
    simu1.error_bar(time_array_err, mean_msd_err, error_size_err)
    plt.show()
    
def msd_error_bar():
    W = 10 # number of trajectories
    N = 10000 # number of steps
    barrier = 2 #kbT
    time_skip = 10
    msd_matrix = simu1.msd_in_matrix(W, N, barrier,time_end = 1, time_skip = 100)
    msd_mean = np.mean(msd_matrix, axis=0)
    print(msd_mean)
    msd_std = np.std(msd_matrix, axis=0)

    time_array = np.arange(0, len(msd_mean), 1)*time_skip*simu1.dt_s 
    msd_mean= msd_mean[:-1]
    time_array = time_array[:-1]
    nb_error_bar = 20
    L = len(time_array)
    time_array_err = []
    msd_std_err = []
    msd_mean_err = []

    for i in range(1,nb_error_bar):
        time_array_err.append(time_array[L*i//nb_error_bar])
        msd_mean_err.append(msd_mean[L*i//nb_error_bar])
        msd_std_err.append(msd_std[L*i//nb_error_bar])
    
    #fitting 
    D_opt = simu1.fit_super_diffusion(time_array, msd_mean)
    plt.plot(time_array,simu1.super_diffusion(time_array, D_opt),color = 'red')
    plt.plot(time_array, msd_mean)
    plt.xlabel('time [s]')
    plt.ylabel('mean square displacement [rad^2]')
    simu1.error_bar(time_array_err, msd_mean_err, msd_std_err)
    plt.show()

print(simu1.dt_s)
msd_error_bar()
