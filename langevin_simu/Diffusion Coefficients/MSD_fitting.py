from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from class_diffusion_potential import *

simu1 = DiffusionSimulation(frequency = 26,torque  = 0 )
simu2 = DiffusionSimulation(frequency = 26,torque  = 100 )
def msd_with_std_no_torque(simu,barrier):
    W = 10 # number of trajectories
    N = 100000 # number of steps
    time_skip = 100
    msd_matrix = simu.msd_in_matrix(W, N, barrier,time_end = 1/4, time_skip = time_skip)
    msd_mean = np.mean(msd_matrix, axis = 0)
    msd_std = np.std(msd_matrix, axis=0)
    time_array = np.arange(1, len(msd_mean) + 1) * simu.dt_s * time_skip
    einstein_diff =simu.slope_fit(time_array,msd_mean)
    D_LJ = simu1.lifson_jackson_noforce(barrier)
    print(type(D_LJ),D_LJ)
    plt.plot(time_array, 2*einstein_diff * time_array, label = f'fit, D_fit = {einstein_diff:.2f},D_Lifson_Jackson = {D_LJ:.2f}, D_einstein = {simu.rotational_einstein_diff:.2f},barrier amplitude in kT = {barrier:.2f}, torque = {simu.torque:.2f}, msd points : {len(msd_mean)}')
    plt.fill_between(time_array, msd_mean + msd_std, msd_mean -msd_std, color='gray', alpha=0.5, label='Std')
    plt.scatter(time_array, msd_mean, label = f'Mean MSD, amplitude =  {barrier:.2f}', s = 0.5)
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel(f'mean square displacement [rad^2]')
    
"""     
for i in range (0,100,20):
    traj = np.unwrap(simu1.static_process(10000, i))
    plt.plot(np.arange(0, len(traj)) * simu1.dt_s, traj, label = 'amplitude [kT]= ' + str(i))
    plt.xlabel('time [s]')
    plt.ylabel('angular displacement')
    plt.legend()
 """

for i in range (0,5,1):
    msd = msd_with_std_no_torque(simu1,i)
    plt.legend()


def msd_with_std_with_torque(torque,barrier):
    simu = DiffusionSimulation(frequency = 26,torque  = torque )
    W = 10 # number of trajectories
    N = 100000 # number of steps
    time_skip = 100
    msd_matrix = simu.msd_in_matrix(W, N, barrier,time_end = 1/4, time_skip = time_skip)
    msd_mean = np.mean(msd_matrix, axis = 0)
    msd_std = np.std(msd_matrix, axis=0)
    time_array = np.arange(1, len(msd_mean) + 1) * simu.dt_s * time_skip
    D_fit,D_std = simu.fit_super_diffusion(time_array, msd_mean)
    return time_array,msd_mean,msd_std,D_fit,D_std


#msd_with_std_no_torque(simu1,barrier = 0)
torque_input0 = 0
barrier_input0 = 0

""" for j in np.arange(0,1000,100):
    time_array,msd_mean,msd_std,D_fit,D_std = msd_with_std_with_torque(torque = j, barrier = barrier_input0 )
    #plt.plot(time_array, simu.super_diffusion(time_array, D_fit), color = 'r', label = f'fit, barrier amplitude in kT = {barrier:.2f}, torque = {simu.torque:.2f}, msd points : {len(msd_mean)}')
    #plt.fill_between(time_array, msd_mean + msd_std, msd_mean -msd_std, color='gray', alpha=0.5, label='Std')
    plt.scatter(time_array, msd_mean, label = f'Mean MSD with torque = {j:.2f}', s = 0.5)
    plt.xlabel('time [s]')
    plt.ylabel(f'mean square displacement [rad^2], ')
    plt.legend()
 """
plt.show()