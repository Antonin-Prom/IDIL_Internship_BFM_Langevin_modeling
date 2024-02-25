import numpy as np
from Class_particle_in_potential import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit


einstein_diff = 0.1648


data = np.load('msd_no_torque.npy',allow_pickle=True)
loaded_dict = data[()]
time = loaded_dict['time']
std = loaded_dict['std']
msd = loaded_dict['msd']
D_eff = loaded_dict['D_eff']



def theory_plot():
    sim = DiffusionSimulation()
    t_arr = np.arange(0,10^3,0.01)
    theo_msd = sim.theory_curve(t_arr,2)/(26*26)
    plt.loglog(t_arr,theo_msd)


def plot_potential_no_torque(index):
    #plt.loglog(time[index],msd[index],label= f'Simulated MSD, Einstein diffusion = {einstein_diff:.3f}')
    #plt.fill_between(time[index], msd[index] - std[index], msd[index] + std[index], color='gray', alpha=0.3, label='standard deviation')
    plt.xlabel('t[s]')
    plt.ylabel('MSD [rad²]')
    plt.legend()
    plt.show() 

theory_plot()
plt.show()
