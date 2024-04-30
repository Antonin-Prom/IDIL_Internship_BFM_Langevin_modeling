from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_v2_periodic_potential import *
from scipy.stats import norm
from scipy.optimize import curve_fit


traj_matrix = []
def generate_mean_traj(N,L,A,torque):
    part = DiffusionSimulation(dt = 1e-5,torque = 100)
    for i in range(N):
        traj = part.proceed_traj1(L ,A)
        traj_matrix.append(traj)
    mean_traj = np.mean(traj_matrix,axis=0)
    return mean_traj

#mean_traj = generate_mean_traj(1,1000000,3,100)


def get_trajs(amplitude,dt):
    trajs = np.load(f'trajectories_10000000points_amplitude_{amplitude}kT_dt_{dt}_torque_0kT.npy')                  
    return trajs

def hist_at_t(amplitude,dt,t):
    trajs = get_trajs(amplitude,dt)
    matrix_t = [trajs[i][t]  for i in range(len(trajs))]
    return matrix_t


def total_hist(amplitude,dt):
    """
    Combine the whole histogram of every traj in a single one : meaningless

    Parameters
    ----------
    amplitude : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    trajs = get_trajs(amplitude,dt)
    num_bins = 10000
    accumulated_hist = np.zeros(num_bins)
    for traj in trajs:
        hist, _ = np.histogram(traj, bins=num_bins,range=(-np.pi, np.pi))
        accumulated_hist += hist
    
    mean_hist = accumulated_hist*num_bins / (len(trajs)*len(trajs[0]))
    bins_space = np.linspace(-np.pi,np.pi,num_bins)
    plt.plot(bins_space,mean_hist)
    plt.xlabel('angle [rad]')
    plt.ylabel('density')
    plt.show()
    
def hist_at_t(t):
    trajs = np.load('trajectories_1000000,nb_traj_1000points_amplitude_6kT,frequency_10_dt_0.001_torque_0kT.npy')
    matrix_t = [trajs[i][t]  for i in range(len(trajs))]
    return matrix_t

def plot_hist(t):
    plt.hist(hist_at_t(t),bins = 200,density=True)
    plt.title(f'Histogram of positions at t={t}ms over 1000 realisations')
    plt.xlabel('angle [rad]')
    plt.ylabel('density')
    plt.show()
    
# A faire: 10000 realisations à 2kT, trouver comment fixer la densité à 1
    
