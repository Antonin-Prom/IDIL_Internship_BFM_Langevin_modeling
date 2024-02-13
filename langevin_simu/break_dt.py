from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_in_potential import *


dt_list = [1e-8,1e-7,1e-6,1e-5,1e-4]
def break_simu():
    for dt in dt_list:
        simu = DiffusionSimulation(frequency = 26,torque  = 10,dt = dt )
        N = int(3/dt) # Number of step
        time_skip = int(N/10)
        msd_matrix = simu.msd_in_matrix(10, N, 0,time_end = 1/4, time_skip = time_skip)
        msd_mean = np.mean(msd_matrix, axis = 0)
        time_array = np.arange(1, len(msd_mean) + 1) * simu.dt_s * time_skip
        plt.plot(time_array, msd_mean, label = f'Dt = {dt}')
        plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('mean square displacement [rad^2]')
    plt.show()

break_simu()
