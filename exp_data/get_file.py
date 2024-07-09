import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from  Langevin_Class_v3 import *
viridis = plt.cm.viridis


def load():
    data = torch.load('exp_data/trajectory.pt')
    return data

def traj_array():
    tracking_data = load()
    x = tracking_data['cx'].numpy()
    y = tracking_data['cy'].numpy()
    return x,y

def plot():
    x,y = traj_array()
    plt.scatter(x[600000:708397],y[600000:708397],s=1)
    plt.show()

def convert_to_angle(x,y):
    return np.arctan2(x,y)

def mean_square_displacement(array):
    traj = np.unwrap(array)
    msd = []
    centered_array = traj - np.mean(traj)
    for j in range(1,int(len(centered_array))):
        M = np.mean((centered_array[:-j] - centered_array[j:])**2)
        msd.append(M)
    return np.array(msd)

def slope_fit(t,msd):
    slope, _ = np.polyfit(t, msd, 1)
    diffusion = slope/2
    return diffusion

def super_diffusion(x, D, alpha):
    return 2*D * x**alpha

def fit_super_diffusion(x_data, y_data):
    params, _ = curve_fit(super_diffusion, x_data, y_data)
    D_opt, alpha_opt = params
    return D_opt, alpha_opt



#exp_parameter:
T_K = 273 + 22
FPS = 2000
dt_s = 1/FPS
space_ratio = 0.095e-6 #m/px
colors = viridis(np.linspace(0, 1, 10))
p = LangevinSimulator(dt=1e-5)

# initialising array
def exp_traj():

    x,y = traj_array()
    x,y = x[600000:708397]*space_ratio,y[600000:708397]*space_ratio
    shift_x,shift_y = np.mean(x),np.mean(y)
    x,y = x - shift_x,y - shift_y
    theta_traj = convert_to_angle(x,y)
    t = np.arange(0,len(theta_traj),1)*dt_s
    return t,theta_traj,x,y

def plot_traj():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = 0
    t,theta_traj,x,y = exp_traj()
    plt.scatter(x,y,s=0.01,color=colors[6])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_old_msd():
# diffusion coefficient
    msd = mean_square_displacement(theta_traj)
    #remove_end_aberation
    msd = msd[:-int(len(msd)/20)]
    t = np.arange(0,len(msd),1)*dt_s
    diffusion,alpha = fit_super_diffusion(t,msd)
    plt.scatter(t,msd,s=1, label = f'Diffusion = {diffusion:.2f}, MSD')
    plt.plot(t,super_diffusion(t, diffusion, alpha), label = f'Diffusion = {diffusion:.2f} [rad^2.s⁻1],alpha = {alpha:.2f}, MSD',color = 'red')
    print("Diffusion_coefficient = ",diffusion,'alpha = ',alpha)
    plt.set_xlabel('t[s]')
    plt.set_ylabel('MSD [rad^2]')
    plt.legend()

def exp_msd_removed_mean(plot=True):
    p = LangevinSimulator(dt=dt_s)
    t,traj,x,y = exp_traj()
    
    t = np.arange(len(traj))*p.dt
    traj = traj - t*traj[-1]/t[-1] # Removing the mean speed
    msd_nbpt = 500
    time_end = 1/4
    msd = p.msd(traj, time_end = time_end, msd_nbpt = msd_nbpt, print_time=False)
    max_lagtime = int(len(traj) * time_end)
    total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])*p.dt
    if plot:
        p.configure_plots()
        plt.loglog(total_lag_time,msd)
        plt.xlabel('lag_time [s]')
        plt.ylabel('MSD [rad²]')
        plt.show()

    plt.show() 


def exp_msd_removed_mean_simulation_compared(plot=True):
    """
    TO DO :
    - Produce traj with removed mean with low amplitude and sufficient torque (V0 = 1kT, torque = 15kT ?)
    - compute msd over simulation and overlap the exp_msd with it
    - fit D
    """
    p = LangevinSimulator(dt=dt_s)
    t,traj,x,y = exp_traj()
    
    t = np.arange(len(traj))*p.dt
    traj = traj - t*traj[-1]/t[-1] # Removing the mean speed
    msd_nbpt = 500
    time_end = 1/4
    msd = p.msd(traj, time_end = time_end, msd_nbpt = msd_nbpt, print_time=False)
    max_lagtime = int(len(traj) * time_end)
    total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])*p.dt
    if plot:
        p.configure_plots()
        plt.loglog(total_lag_time,msd)
        plt.xlabel('lag_time [s]')
        plt.ylabel('MSD [rad²]')
        plt.show()

    plt.show() 

#plot_traj()
new_msd_removed_mean(plot=True)

