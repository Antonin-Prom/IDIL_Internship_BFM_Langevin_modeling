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
time_skip = 100  
W = 20
N = 500000
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
    amp,D,_ = LJ_vs_barrier(N) #sim.LJ_mean_denominator(amplitude_array = np.linspace(0, 1e-3, N))
    einstein = np.ones(N)*sim.rotational_einstein_diff
    plt.plot(amp,einstein,label = f'Einstein Diffusion Coefficient = {einstein_diff:.3f}')
    plt.plot(amp,D,label='Lifson & Jackson Diffusion Coefficient')
    plt.xlabel('Barrier Amplitude [kT]')
    plt.ylabel('Effective Diffusion coefficient [rad²/s]')
    plt.title('MSD : free diffusion')
    plt.legend()
        

def diffusion_potential_no_torque(amplitude):
    sim = DiffusionSimulation(dt=1e-5)
    msd_matrix = sim.msd_in_matrix(W, N, amplitude, time_end, time_skip)
    mean_msd = np.mean(msd_matrix,axis=0)
    std = np.std(msd_matrix,axis=0)
    std[0] = std[1]
    t = np.arange(1,len(mean_msd)+1)*time_skip*sim.dt_s
    D_LJ = sim.lifson_jackson_noforce(amplitude)
    popt, pcov = scipy.optimize.curve_fit(linear_D, t, mean_msd, sigma = std, p0 = einstein_diff)
    return t,mean_msd,popt[0],std,D_LJ


def plot_diffusion_potential_no_torque(amplitude):
    t,msd,D,std,D_LJ = diffusion_potential_no_torque(amplitude)
    plt.plot(t,msd,label= f'simulated msd, number of msd points = {len(t):.0f}, trajectory length = {N:.0f}')
    fit = linear_D(t, D)
    plt.plot(t, fit, label= f' 2Dt, fitted diffusion coefficient = {D:.5f}, L&J Diffusion coefficient = {D_LJ:.3f}')
    plt.fill_between(t, msd - std, msd + std, color='gray', alpha=0.3, label='std')
    plt.xlabel('t[s]')
    plt.ylabel('[rad²]')
    plt.title(f'MSD : Potential no torque, amplitude = {amplitude:.3f} kT')
    plt.legend()
    plt.show()
    
#plot_diffusion_potential_no_torque()

def diffusion_no_potential_torque():
    sim = DiffusionSimulation(dt=1e-6, torque = 10)
    amplitude = 0
    msd_matrix = sim.msd_in_matrix(W, N, amplitude, time_end, time_skip)
    mean_msd = np.mean(msd_matrix,axis=0)
    std = np.std(msd_matrix,axis=0)
    std[0] = std[1]
    t = np.arange(0,len(mean_msd))*time_skip*sim.dt_s
    popt, pcov = scipy.optimize.curve_fit(linear_D, t, mean_msd, sigma = std, p0 = einstein_diff)
    D_LJ = sim.lifson_jackson_force(amplitude)
    torque = sim.torque
    return t,mean_msd,popt[0],std,D_LJ,torque

def plot_diffusion_no_potential_torque():
    t,msd,D,std,D_LJ,torque = diffusion_potential_no_torque()
    plt.plot(t,msd,label= f'simulated msd, number of msd points = {len(t):.0f}, trajectory length = {N:.0f}')
    fit = linear_D(t, D)
    plt.plot(t, fit, label= f' 2Dt, fitted diffusion coefficient = {D:.3f}, L&J Diffusion coefficient = {D_LJ:.3f}')
    plt.fill_between(t, msd - std, msd + std, color='gray', alpha=0.3, label='std')
    plt.xlabel('t[s]')
    plt.ylabel('[rad²]')
    plt.title(f'MSD : No potential, torque = {torque:.3f} kT')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
def diffusion_no_potential_torque():
    sim = DiffusionSimulation(dt=1e-6, torque = 10)
    amplitude = 0
    msd_matrix = sim.msd_in_matrix(W, N, amplitude, time_end, time_skip)
    mean_msd = np.mean(msd_matrix,axis=0)
    std = np.std(msd_matrix,axis=0)
    std[0] = std[1]
    t = np.arange(0,len(mean_msd))*time_skip*sim.dt_s
    popt, pcov = scipy.optimize.curve_fit(linear_D, t, mean_msd, sigma = std, p0 = einstein_diff)
    D_LJ = sim.lifson_jackson_force(amplitude)
    return t,mean_msd,popt[0],std,D_LJ

def plot_diffusion_no_potential_torque():
    t,msd,D,std,D_LJ = diffusion_potential_no_torque()
    plt.plot(t,msd,label= f'simulated msd, number of msd points = {len(t):.0f}, trajectory length = {N:.0f}')
    fit = linear_D(t, D)
    plt.plot(t, fit, label= f' 2Dt, fitted diffusion coefficient = {D:.3f}, L&J Diffusion coefficient = {D_LJ:.3f}')
    plt.fill_between(t, msd - std, msd + std, color='gray', alpha=0.3, label='std')
    plt.xlabel('t[s]')
    plt.ylabel('[rad²]')
    plt.legend()
    plt.show()


