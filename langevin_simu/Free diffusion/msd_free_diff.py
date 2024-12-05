# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:35:33 2024

@author: CBS
"""

import numpy as np
from Class_particle_v2_free_diffusion import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit


p0 = DiffusionSimulation2()
D = p0.rotational_einstein_diff

def parabolic_drift(t):
    return 2*D*t 

def plot_drift_msd_std():
    traj_drift = np.load('trajectories_1000000points_amplitude_0kT_dt_1e-05_torque_20kT.npy')
    p = DiffusionSimulation2(torque = 20)
    time_d,mean_msd_d,std,msd_matrix_d = p.mean_msd_and_time_axis(traj_drift,nb_chunks=10, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = 10)
    time_d *= 1e-5
    v_eff = p.torque/p.rotational_drag
    popt, pcov = scipy.optimize.curve_fit(parabolic_drift, time_d, mean_msd_d)
    residuals = mean_msd_d - parabolic_drift(time_d, *popt)
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((mean_msd_d - np.mean(mean_msd_d))**2)
    R_squared = 1 - (SS_res / SS_tot)
    print(R_squared)
    plt.plot(time_d,mean_msd_d, label = 'Numerical MSD, torque = 20kT')
    plt.plot(time_d, parabolic_drift(time_d, *popt), linestyle = '--',color = 'black', label=f'Fitted Curve,v_fit = {popt[0]:.3f}')
    plt.plot(time_d, parabolic_drift(time_d, v_eff), linestyle = '--',color = 'red', label=f'Theoretical MSD: v_eff = {v_eff:.3f}')

    plt.fill_between(time_d, mean_msd_d -std, mean_msd_d + std, color='gray', alpha=0.6)
    plt.xlabel('Lag time (s)')
    plt.ylabel('MSD (rad²)')
    plt.legend()
    plt.show()





