from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_in_potential import *


<<<<<<< Updated upstream
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
=======
dt_list = [1e-8,1e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3]
msd_means_list = []
time_arrays_list = []

def break_simu():
    for dt in dt_list:
        simu = DiffusionSimulation(frequency=26, torque=10, dt=dt)
        N = int(1 / dt)  # Number of steps
        time_skip = int(N / 10)
        
        msd_matrix = simu.msd_in_matrix(10, N, 0, time_end=1/4, time_skip=time_skip)
        msd_mean = np.mean(msd_matrix, axis=0)
        
        # Reduce the size of time_array by using a smaller step
        time_array = np.arange(1, len(msd_mean) + 1, 2).astype(np.float32) * simu.dt_s * time_skip
        
        msd_means_list.append(msd_mean)
        time_arrays_list.append(time_array)

    np.savez("msd_data.npz", msd_means_list=msd_means_list, time_arrays_list=time_arrays_list)

data = np.load("msd_data.npz")

# Access the arrays
msd_means_list = data['msd_means_list']
time_arrays_list = data['time_arrays_list']
dt_list = [1e-8,1e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3]
print(msd_means_list[0])
# Plot MSD for each dt
for i in range(len(dt_list)):
    plt.plot(time_arrays_list[0][0], msd_means_list[i][0], label=f'dt = {dt_list[i]:.0e}')

plt.xlabel('Time')
plt.ylabel('MSD Mean')
plt.title('Mean Squared Displacement (MSD) for Different dt')
plt.legend()
plt.show()


>>>>>>> Stashed changes
