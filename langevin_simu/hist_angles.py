from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_in_potential import *



traj_matrix = []
def generate_mean_traj(N,L,A,torque):
    part = DiffusionSimulation(dt = 1e-5,torque = 100)
    for i in range(N):
        traj = part.proceed_traj1(L ,A)
        traj_matrix.append(traj)
    mean_traj = np.mean(traj_matrix,axis=0)
    return mean_traj

mean_traj = generate_mean_traj(1,1000000,0.5,0)


def get_trajs(amplitude,dt):
    trajs = np.load(f'trajectories_100000000points_amplitude_{amplitude}kT_dt_{dt}.npy')
    return trajs

def plot_hist(amplitude,dt,i):
    trajs = get_trajs(amplitude,dt)
    traj = trajs[i]
    plt.hist(traj,bins = 2600,density=True)
    plt.xlabel('angle [rad]')
    plt.ylabel('density')
    plt.show()

