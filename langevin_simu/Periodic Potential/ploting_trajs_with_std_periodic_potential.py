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

    
    
def plot_trajs():
    trajs = [np.load(f'trajectories_1000000points_amplitude_0kT_dt_1e-05_torque_0kT.npy')[0],np.load(f'trajectories_1000000points_amplitude_1kT_dt_1e-05_torque_0kT.npy')[0],np.load(f'trajectories_1000000points_amplitude_2kT_dt_1e-05_torque_0kT.npy')[0],
             np.load(f'trajectories_1000000points_amplitude_3kT_dt_1e-05_torque_0kT.npy')[0]]
    
    t = np.arange(0,len(trajs[0]))*1e-5
    plt.plot(t[: ],trajs[0][: ],linewidth =0.5,label='0kT')
    plt.plot(t[: ],trajs[1][: ],linewidth =0.5,label='1kT')
    plt.plot(t[: ],trajs[2][: ],linewidth =0.5,label='2kT')
    plt.plot(t[: ],trajs[3][: ],linewidth =0.5,label='3kT')
    plt.xlabel('time [s]')
    plt.ylabel('displacement [rad]')
    plt.legend()
    plt.show()

plot_trajs()
