from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from class_diffusion_potential import *

simu1 = DiffusionSimulation(frequency = 26,torque  = 0 )
simu2 = DiffusionSimulation(frequency = 26,torque  = 100 )
def msd_with_std_no_torque(simu,barrier):
    W = 10 # number of trajectories
    N = 10000 # number of steps
    time_skip = 10
    msd_matrix = simu.msd_in_matrix(W, N, barrier,time_end = 1/4, time_skip = time_skip)
    msd_mean = np.mean(msd_matrix, axis = 0)
    msd_std = np.std(msd_matrix, axis=0)
    time_array = np.arange(1, len(msd_mean) + 1) * simu.dt_s * time_skip
    einstein_diff =simu.slope_fit(time_array,msd_mean)
    #plt.plot(time_array, 2*einstein_diff * time_array, 'r', label = f'fit, D_fit = {einstein_diff:.2f}, D_theory = {simu.rotational_einstein_diff:.2f}, D_fit/D_theory = {einstein_diff/simu.rotational_einstein_diff:.2f},barrier amplitude in kT = {barrier:.2f}, torque = {simu.torque:.2f}, msd points : {len(msd_mean)}')
    #plt.fill_between(time_array, msd_mean + msd_std, msd_mean -msd_std, color='gray', alpha=0.5, label='Std')
    plt.scatter(time_array, msd_mean, label = f'Mean MSD, amplitude =  {barrier:.2f}', s = 0.5)
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel(f'mean square displacement [rad^2]')
    
    
for i in range (0,1000,100):
    traj = np.unwrap(simu1.static_process(1000, i))
    plt.plot(np.arange(0, len(traj)) * simu1.dt_s, traj, label = 'amplitude [kT]= ' + str(i))
    plt.legend()
plt.show()

def msd_with_std_with_torque(simu):
    W = 50 # number of trajectories
    N = 100000 # number of steps
    barrier = 0 #kbT
    time_skip = 100
    msd_matrix = simu.msd_in_matrix(W, N, barrier,time_end = 1/4, time_skip = time_skip)
    msd_mean = np.mean(msd_matrix, axis = 0)
    msd_std = np.std(msd_matrix, axis=0)
    time_array = np.arange(1, len(msd_mean) + 1) * simu.dt_s * time_skip
    einstein_diff =simu.slope_fit(time_array,msd_mean)
    plt.plot(time_array, 2*einstein_diff * time_array, 'r', label = f'fit, D_fit = {einstein_diff:.2f}, D_theory = {simu.rotational_einstein_diff:.2f}, D_fit/D_theory = {einstein_diff/simu.rotational_einstein_diff:.2f}, barrier amplitude in kT = {barrier:.2f}, torque = {simu.torque:.2f}, msd points : {len(msd_mean)}')
    plt.fill_between(time_array, msd_mean + msd_std, msd_mean -msd_std, color='gray', alpha=0.5, label='Std')
    plt.scatter(time_array, msd_mean, color = 'black', label = 'Mean MSD', s = 0.5)
    plt.xlabel('time [s]')
    plt.ylabel(f'mean square displacement [rad^2], ')
    plt.legend()
    plt.show()

#msd_with_std_no_torque(simu1)
