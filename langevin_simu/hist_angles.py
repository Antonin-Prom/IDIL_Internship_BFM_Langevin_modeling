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

#mean_traj = generate_mean_traj(1,1000000,3,100)


def get_trajs(amplitude,dt):
    trajs = np.load(f'trajectories_1000000points_amplitude_{amplitude}kT_dt_{dt}_torque_0.npy')                  
    return trajs

def hist_at_t(amplitude,dt,t):
    trajs = get_trajs(amplitude,dt)
    matrix_t = [trajs[i][t]  for i in range(len(trajs))]
    return matrix_t


def plot_hist(amplitude,dt,i):
    trajs = get_trajs(amplitude,dt)
    traj = trajs[i]
    plt.hist(traj,bins = 2600,density=True)
    plt.xlabel('angle [rad]')
    plt.ylabel('density')
    plt.show()

plt.hist(matrix_t,bins = 10000,density=True)
plt.title('Histogram of positions at t=0 over 10000 realisations')
plt.xlabel('angle [rad]')
plt.ylabel('density')
plt.show()