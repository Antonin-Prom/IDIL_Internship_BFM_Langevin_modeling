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
"""
trajs5 = np.load(f'trajectories_100000points_amplitude_0kT_dt_1e-05_torque_0.npy')
std5 = np.std(trajs5, axis=0)
"""

D = 0.16480238117707385

    
    
def plot_trajs(trajs,std,mean):
    torque = 2.0709678000000002e-20 #5kT
    drag = 2.5132741228718344e-20
    v_eff = -torque/drag
    
    trajs = trajs[:20]
    t = np.arange(len(trajs[0]))*1e-5
    for i in range(len(trajs)):
        plt.plot(t,trajs[i],linewidth =0.5)
    coefficients = np.polyfit(t, mean, 1)
    slope = coefficients[0]
    r2 = (v_eff - slope)**2
    plt.fill_between(t, mean -std, mean + std, color='gray', alpha=0.6, label='std')
    plt.plot(t, mean,linestyle = '--',color = 'black',label=f'mean, (v_eff - mean_slope_fit)^2 = {r2:.5f} ')
    plt.xlabel('time [s]')
    plt.ylabel('displacement [rad]')
    plt.legend()
    plt.show()
