# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:53:47 2024

@author: CBS
"""

import numpy as np
from Class_particle_in_potential import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit


time_end = 1/4
time_skip = 10
W = 100
N = 100000
einstein_diff = 0.1648

def linear_D(t,D):
    return 2*D*t

def free_diffusion_weighted_fit():
    sim = DiffusionSimulation(dt=1e-6)
    amplitude = 0
    msd_matrix = sim.msd_in_matrix(W, N, amplitude, time_end, time_skip)
    mean_msd = np.mean(msd_matrix,axis=0)
    std = np.std(msd_matrix,axis=0)
    std[0] = std[1]
    t = np.arange(0,len(mean_msd))*time_skip*sim.dt_s

    popt, pcov = scipy.optimize.curve_fit(linear_D, t, mean_msd, sigma = std, p0 = einstein_diff, bounds = (einstein_diff-0.1*einstein_diff,einstein_diff+0.1*einstein_diff))
    return t,mean_msd,popt[0],std

def free_diffusion_weighted_fit1():
    sim = DiffusionSimulation(dt=1e-6)
    amplitude = 0
    mean_msd,std = sim.mean_msd_with_theoretical_variance(W, N, amplitude, time_end, time_skip)
    t = np.arange(0,len(mean_msd))*time_skip*sim.dt_s
    popt, pcov = scipy.optimize.curve_fit(linear_D, t, mean_msd, sigma = std, p0 = einstein_diff)
    return t,mean_msd,popt[0],std


def plot_free_Diffusion():
    t,msd,D,std = free_diffusion_weighted_fit()
    print(len(t),len(msd))
    plt.plot(t,msd,label= f'Simulated MSD, Einstein diffusion = {einstein_diff:.3f}')
    fit = linear_D(t, D)
    plt.plot(t, fit, label= f' y = 2Dt, Fitted diffusion coefficient = {D:.3f}')
    plt.fill_between(t, msd - std, msd + std, color='gray', alpha=0.3, label='standard deviation')
    plt.xlabel('t[s]')
    plt.ylabel('MSD [rad²]')
    plt.legend()
    plt.show()
    

def LJ_vs_barrier(N):
    amp = np.linspace(0,0.001,N)
    einstein = np.ones(N)*einstein_diff
    D = np.zeros(N)
    sim = DiffusionSimulation()
    for i,A in zip(range(len(amp)),amp):
        D[i] = sim.lifson_jackson_noforce(A)
    
    return amp,D,einstein

def plot_LJ_barrier(N):
    sim = DiffusionSimulation()
    amp,D = sim.LJ_mean_denominator(amplitude_array = np.linspace(0, 1e-3, N))
    einstein = np.ones(N)*sim.rotational_einstein_diff
    plt.plot(amp,einstein,label = f'Einstein Diffusion Coefficient = {einstein_diff:.3f}')
    plt.plot(amp,D,label='Lifson & Jackson Diffusion Coefficient')
    plt.xlabel('Barrier Amplitude [kT]')
    plt.ylabel('Effective Diffusion coefficient [rad²/s]')
    plt.legend()
        

def diffusion_potential_no_torque():
    sim = DiffusionSimulation(dt=1e-6)
    amplitude = 3
    msd_matrix = sim.msd_in_matrix(W, N, amplitude, time_end, time_skip)
    mean_msd = np.mean(msd_matrix,axis=0)
    std = np.std(msd_matrix,axis=0)
    std[0] = std[1]
    t = np.arange(0,len(mean_msd))*time_skip*sim.dt_s
    popt, pcov = scipy.optimize.curve_fit(linear_D, t, mean_msd, sigma = std, p0 = einstein_diff, bounds = (einstein_diff-0.1*einstein_diff,einstein_diff+0.1*einstein_diff))
    return t,mean_msd,popt[0],std

def plot_diffusion_potential_no_torque():
    t,msd,D,std = free_diffusion_weighted_fit()
    print(len(t),len(msd))
    plt.plot(t,msd,label= f'simulated msd, L&J diffusion = {D:.3f}, number of msd points = {len(t):.0f}, trajectory length = {N:.0f}')
    fit = linear_D(t, D)
    plt.plot(t, fit, label= f' 2Dt, fitted diffusion coefficient = {D:.3f},')
    plt.fill_between(t, msd - std, msd + std, color='gray', alpha=0.3, label='std')
    plt.xlabel('t[s]')
    plt.ylabel('[rad²]')
    plt.legend()
    plt.show()
