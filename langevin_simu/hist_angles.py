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

mean_traj = generate_mean_traj(1,1000000,3,100)
plt.hist(mean_traj,bins = 2600,density=True)
plt.xlabel('angle [rad]')
plt.ylabel('density')
plt.show()

