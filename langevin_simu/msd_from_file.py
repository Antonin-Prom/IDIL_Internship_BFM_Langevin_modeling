import numpy as np
from Class_particle_in_potential import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit


einstein_diff = 0.164848


data = np.load('msd_no_torque.npy',allow_pickle=True)
loaded_dict = data[()]
time = loaded_dict['time']
std = loaded_dict['std']
msd = loaded_dict['msd']
D_eff = loaded_dict['D_eff']



def theory_plot():
    sim = DiffusionSimulation()
    t_arr = np.arange(36000000)*1e-4
    theo_msd = sim.theory_curve_oscillatory(t_arr,2)/(26*26)
    popt, pcov = scipy.optimize.curve_fit(linear_D, t_arr, theo_msd)
    D_fit,shift = popt[0],popt[1]
    fit = linear_D(t_arr, D_fit,shift)
    
    plt.loglog(t_arr,theo_msd, label = 'theory defaveri')
    plt.plot(t_arr, fit, label= f' y = 2Dt, Fitted diffusion coefficient = {D_fit:.3f}, D_LJ = {D_eff[0]:.3f}')
    plt.xlabel('Dt/a²')
    plt.ylabel('MSD')
    
def linear_D(t,D,shift):
    return 2*D*t + shift

def plot_potential_no_torque(index):
    x_axis = einstein_diff*time[index]/(26*26)
    plt.plot(x_axis,msd[index],label= f'Simulated MSD, Barrier = 2kT, Einstein diffusion = {einstein_diff:.3f}')
    #plt.fill_between(x_axis, msd[index] - std[index], msd[index] + std[index], color='gray', alpha=0.3, label='standard deviation')
    popt, pcov = scipy.optimize.curve_fit(linear_D, time[index], msd[index],sigma = std[index])
    D_fit,shift = popt[0],popt[1]
    fit = linear_D(time[index], D_fit,shift)
    plt.plot(x_axis, fit, label= f' y = 2Dt, Fitted diffusion coefficient = {D_fit:.3f}, D_LJ = {D_eff[index]:.3f}')
    print(popt,D_eff[index])
    plt.xlabel('Dt/freq²')
    plt.ylabel('MSD [rad²]')
    plt.legend()
    plt.show() 

theory_plot()
plt.show()
