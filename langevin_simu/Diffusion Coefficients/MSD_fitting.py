from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_in_potential import *

simu1 = DiffusionSimulation(frequency = 26,torque  = 0 )
simu2 = DiffusionSimulation(frequency = 26,torque  = 100 )
def msd_with_std_no_torque(simu,barrier):
    """Compute msd with standard deviation for a simulation with no torque

    Args:
        simu (object): particle from the Class
        barrier (float): barrier amplitude

    Returns:
        array: time
        array: mean of mean square displacement
        array: standard deviation
        float: fit coefficient
        float: theoretical coefficient
    """
    W = 10 # number of trajectories
    N = 100000 # number of steps
    time_skip = 100
    msd_matrix = simu.msd_in_matrix(W, N, barrier,time_end = 1/4, time_skip = time_skip)
    msd_mean = np.mean(msd_matrix, axis = 0)
    msd_std = np.std(msd_matrix, axis=0)
    time_array = np.arange(1, len(msd_mean) + 1) * simu.dt_s * time_skip
    D_fit =simu.slope_fit(time_array,msd_mean)
    D_LJ = simu1.lifson_jackson_noforce(barrier)
    return time_array,msd_mean, msd_std, D_fit, D_LJ

def plot_msd_no_torque(simu,barrier):
    time_array,msd_mean, msd_std, D_fit, D_LJ = msd_with_std_no_torque(simu,barrier)
    plt.plot(time_array, 2*D_fit * time_array, label = f'fit, D_fit = {D_fit:.3f},D_Lifson_Jackson = {D_LJ:.3f}, D_einstein = {simu.rotational_einstein_diff:.3f},barrier amplitude in kT = {barrier:.3f}, torque = {simu.torque:.3f}, msd points : {len(msd_mean)}')
    plt.fill_between(time_array, msd_mean + msd_std, msd_mean -msd_std, color='gray', alpha=0.5, label='Std')
    plt.scatter(time_array, msd_mean, label = f'Mean MSD, amplitude =  {barrier:.3f}', s = 0.5)
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel(f'mean square displacement [rad^2]')
    
def plot_multiple_msd_no_torque(simu):
    for i in range (0,5,1):
        plot_msd_no_torque(simu,i)
        
def msd_with_std_with_torque(torque,barrier):
    
    """Compute the mean square displacement for a simulation with torque

    Args:
        torque (float): torque applied to the particle
        barrier (float): barrier amplitude

    Returns:
        object: particle from the Class
        array: time
        array: mean of mean square displacement
        array: standard deviation

    """
    simu = DiffusionSimulation(frequency = 26,torque  = torque )
    W = 50 # number of trajectories
    N = 100000 # number of steps
    time_skip = 100
    msd_matrix = simu.msd_in_matrix(W, N, barrier,time_end = 1/4, time_skip = time_skip)
    msd_mean = np.mean(msd_matrix, axis = 0)
    msd_std = np.std(msd_matrix, axis=0)
    time_array = np.arange(1, len(msd_mean) + 1) * simu.dt_s * time_skip
    return simu,time_array,msd_mean,msd_std

def plot_msd_with_torque(torque,barrier):
    simu,time_array,msd_mean,msd_std = msd_with_std_with_torque(torque,barrier)
    a,b,c = simu.parabola_fit(time_array,msd_mean)
    plt.plot(time_array, a* time_array**2 + b*time_array + c, label = f'parabola_fit, a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, D_einstein = {simu.rotational_einstein_diff:.3f},barrier amplitude in kT = {barrier:.3f}, torque = {torque:.3f}, msd points : {len(msd_mean)}')
    plt.fill_between(time_array, msd_mean + msd_std, msd_mean -msd_std, color='gray', alpha=0.5, label='Std')
    plt.scatter(time_array, msd_mean, label = f'Mean MSD, amplitude =  {barrier:.3f}', s = 0.5)
    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel(f'mean square displacement [rad^2]')
    
def plot_compare_torque_barrier(torque):
    for i in range (0,5,1):
        plot_msd_with_torque(torque,i)

plt.show()