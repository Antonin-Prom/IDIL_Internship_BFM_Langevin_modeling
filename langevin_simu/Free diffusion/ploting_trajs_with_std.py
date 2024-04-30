# -*- coding: utf-8 -*-
"""
Plots the traj with variance
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import numpy as np
import time

trajs5 = np.load(f'trajectories_100000points_amplitude_0kT_dt_1e-05_torque_0.npy')
std5 = np.std(trajs5, axis=0)


trajs4 = np.load('trajectories_10000points_amplitude_0kT_dt_0.0001_torque_0.npy')
std4 = np.std(trajs4, axis=0)


trajs3 = np.load('trajectories_1000points_amplitude_0kT_dt_0.001_torque_0.npy')
std3 = np.std(trajs3, axis=0)

D = 0.16480238117707385

def plot_stds():
    t = np.arange(len(trajs3[0]))*1e-3
    plt.plot(t,std5[::100],linewidth = 2, alpha = 0.5, color = 'lightblue', label="Standard deviation for $\Delta$t = 10^-5s")
    plt.plot(t,std4[::10],color = 'orange',label="Standard deviation for $\Delta$t = 10^-4s")
    plt.plot(t,std3,color = 'lightgreen', alpha = 1, label="Standard deviation for $\Delta$t = 10^-3s")
    plt.plot(t,np.sqrt(2*D*t), linestyle = '--', color = 'red',label = "f(t) = (2Dt)^(1/2)")
    plt.xlabel('time [s]')
    plt.ylabel('Standard deviation [rad]')
    plt.legend()
    plt.show()
    
    
def plot_trajs(trajs,std):
    trajs = trajs[:20]
    for i in range(len(trajs)):
        t = np.arange(len(trajs[0]))*1e-5
        plt.plot(t,trajs[i],linewidth =0.5)
    plt.fill_between(t, -std, std, color='gray', alpha=0.6, label='std')
    plt.xlabel('time [s]')
    plt.ylabel('displacement [rad]')
    plt.show()
    
plt.plot(T,(T-1)-(T-1)**3)
plt.plot(T,(T-1))