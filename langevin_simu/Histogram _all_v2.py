# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:14:09 2024

@author: CBS
"""
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import time
from scipy.interpolate import interp1d
import scipy
from numba import njit
from matplotlib.lines import Line2D
from scipy.stats import norm
import math

@njit
def _make_trace(x, npts, dUdx, dt, gamma, thermnoise):
    ''' langevin finite difference integrator, called by main_traj_ for numba acceleration '''
    for i in range(npts - 1):
        dUdx_idxi = int(np.round(x[i]*len(dUdx)/(2*np.pi)) - 1)
        force = -dUdx[dUdx_idxi]
        # Langevin eq:
        x[i+1] = x[i] + force*dt/gamma + thermnoise[i]
        x[i+1] = np.mod(x[i+1], 2*np.pi)
    return x


@njit(parallel=True)
def single_msd(traj, total_lag_time):
    msd_results = np.zeros_like(total_lag_time, dtype=np.float64)
    for i in range(len(total_lag_time)):
        lag = total_lag_time[i]
        msd = np.mean((traj[:-lag] - traj[lag:]) ** 2)
        msd_results[i] = msd
    return msd_results

class LangevinSimulator:
     
    def __init__(self,frequency = 10, torque = 0, dt = None, x0 = 0, analytical = True):
        # Constants
        self.KT = 300*1.3806452e-23
        self.R_m = 1e-6
        self.m_kg = 1.1e-14 
        self.viscosity_NS_m2 = 0.001
        #self.gamma = 8 * np.pi * self.viscosity_NS_m2 * self.R_m**3
        self.L = 100e-9   #cylinder_length
        self.gamma = 3.841*np.pi*self.viscosity_NS_m2*self.L*self.R_m**2*(1+0.3) # [Nsm] cylinder gamma_rot_parallel for diam=0.5*length
        self.moment_inertia = (2/5)*self.m_kg*self.R_m**2
        self.tau = self.moment_inertia / self.gamma
        self.D = self.KT / self.gamma
        self.dt = dt # tau
        self.frequency = frequency
        self.space_step = 1e-8
        self.torque = torque
        self.x0 = x0
        self.analytical = analytical
        self.x_pot = np.linspace(0, 2*np.pi, 50000)
    
    
    def make_potential_sin(self, ampl=None, plots=False):
        ''' 
        periodic sinusoidal potential, ampl and tilt in KT units
        return array
        '''
        U = ampl*self.KT*np.cos(self.x_pot*self.frequency) - self.torque*self.x_pot/(2*np.pi)*self.KT
        if plots:
            plt.figure('make_potential_sin', clear=True)
            plt.plot(self.x_pot, U/self.KT, ',')
            plt.ylabel('pot. [KT]')
        return U
    
    def analytical_potential(self,x,A):
        U = A*np.cos(x*self.frequency) - self.torque*x/(2*np.pi)
        return U
    
    def main_traj_(self, N, A, U, x0, ide ):
        """
        Langevin integration using numba. The potential U is an argument of type array.
        U = self.make_potential_sin(self.x_pot,ampl = A)
        """
        dUdx = np.diff(U)/(2*np.pi/len(U))
        x = np.zeros(N) 
        x[0] = x0[ide]
        thermnoise = np.sqrt(2*self.D*self.dt)*np.random.randn(N)
        x = _make_trace(x, N, dUdx, self.dt, self.gamma, thermnoise)   
        return x 
    
    def run_parallel_numba(self, repetitions=None, n_jobs=5, npts = int(1e7), x0 = None, Amplitude = None, torque = 0,iteration = 0, save = False, print_time = False, plots = False):
        ''' 
        parallel computations to generate multiple trajectories in one go
        parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
        '''
        # Parallel:
        if print_time == True:
            print('run_serial_parallel(): Parallel working...')
        t0 = time.time()
        sin_pot = self.make_potential_sin(ampl = Amplitude) 
        parallel_out = Parallel(n_jobs=n_jobs)(delayed(self.main_traj_)(N = npts,A = Amplitude, U = sin_pot, x0 = x0, ide=i) for i in range(repetitions))
        parallel_out = np.asarray(parallel_out, dtype=np.float32)
        if print_time == True:
            print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
        if save == True:
            np.save(f'{iteration}ite_trajectories_{npts:.0f},nb_traj_{repetitions}points_amplitude_{Amplitude}kT,frequency_{self.frequency}_dt_{self.dt}_torque_{torque:.0f}kT',parallel_out)
        if plots:
            U = sin_pot
            x_wrap = parallel_out[0]
            x_unwrap = np.unwrap(x_wrap)
            t0 = time.time()
            idxs = np.linspace(0, npts, 5000, endpoint=False, dtype=int)
            t = np.linspace(0, npts*self.dt, len(idxs), endpoint=False)
            _x_unwrap = x_unwrap[idxs]
            _x_wrap   = x_wrap[idxs]
            fig = plt.figure('make_trace', clear=True, figsize=(12,8))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            ax1.plot(np.linspace(0, 2*np.pi, len(U)), U/self.KT, ',')
            ax1.set_ylabel('Potential [KT]')
            ax1.set_xlabel('angle (rad)')
            ax1.set_title(f'Ampl={Amplitude} KT(pkpk)    dt={self.dt:.1e} s    gamma={self.gamma:.1e} Nsm    D={self.D:.2f} m2/s    torque={self.torque} KT', fontsize=8)
            ax2.plot(t, _x_wrap, ',')
            ax2.set_ylabel('angle (rad)')
            ax2.set_xlabel('time (s)')
            ax3.hist(_x_wrap, 200, orientation='vertical')
            ax3.set_xlabel('angle (rad)')
            ax3.set_ylabel('Occur.')
            ax3.set_xlim([0,2*np.pi])
            ax4.plot(t, _x_unwrap/(2*np.pi), lw=1)
            ax4.set_xlabel('time (s)')
            ax4.set_ylabel('angle (turns)')
            ax4.set_title(f'mean speed: {(np.unwrap(x_wrap)[-1] - np.unwrap(x_wrap)[0])/t[-1]:.3f} Hz', fontsize=8)
            fig.tight_layout()
            print(f'make_trace(): plots done in {time.time() - t0} s.')
        return parallel_out
    
    def chunkify_traj(self, repetitions=None, n_jobs=5, npts = None, x0s = None, Amplitude = None, torque = 0, save = True): #x0 is an array of x0
        '''
        For very large number of points, slice the trajectory generation in chunks 
        '''
        paquet = 1e7
        t0 = time.time()
        print('Chunkify trajectory maker starting...')
        iteration = int(npts/paquet)
        if iteration > 1:
            old_trajs = self.run_parallel_numba(repetitions=repetitions, n_jobs=n_jobs, npts = int(paquet), x0 = x0s, Amplitude = Amplitude, torque = torque)
            for i in range(iteration-1):
                #print(f'iteration n°{i} at {time.time() - t0:.1f} s ')
                all_x0 = [all_x0 := traj[-1] for traj in old_trajs] #getting all the x0
                trajs = self.run_parallel_numba(repetitions=repetitions, n_jobs=n_jobs, npts = int(paquet), x0 = all_x0, Amplitude = Amplitude,  torque = torque, iteration = i,save=False)
                old_trajs = np.concatenate((old_trajs,trajs),axis=1)
        else :
            old_trajs = self.run_parallel_numba(repetitions=repetitions, n_jobs=n_jobs, npts = npts, x0 = x0s, Amplitude = Amplitude, torque = torque)
        print(f'Done in {time.time() - t0:.1f} s ')
        total_len = len(old_trajs[0])
        if save == True:
            np.save(f'final_trajectories_{total_len:.0f},nb_traj_{repetitions}points_amplitude_{Amplitude}kT,frequency_{self.frequency}_dt_{self.dt}_torque_{torque:.0f}kT',old_trajs)
        return old_trajs            
    
    def integrand1(self, x, amplitude):
        return np.exp(-self.analytical_potential(x, amplitude))
    
    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / self.frequency, np.pi / self.frequency, args=(amplitude))
        return result
    
    def lifson_jackson_noforce(self, amplitude): #Meet einstein coeff at 0 barrier
        lifson_jackson1 = self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))
        return lifson_jackson1     
    
    def lifson_jackson_force(self, amplitude, F):
        lifson_jackson2 = (self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))) * ((np.sinh(F*(2*np.pi / self.frequency)/2))/(F*(2*np.pi / self.frequency)/2))**2
        return lifson_jackson2
    



def matrix_at_t(trajs,t):
    m = []
    for traj in trajs:
        traj=np.unwrap(traj)
        m.append(traj[t])
    return m
    
def produce_histograms(npts=1e6,repetitions=1000,free=False,drift=False,periodic=False,tilted_periodic=False,load=False,file_name=None):
    x0 = np.ones(repetitions)*np.pi
    
    time_select = [1*int(npts/10),4*int(npts/10),7*int(npts/10),int(npts -1)]          
    if free == True:
        free = LangevinSimulator(dt=1e-4)
        t0 = time.time()
        if load == False:
            
            trajs = free.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts = int(npts), x0 = x0, Amplitude = 0, torque = 0,iteration = 0, save = False, print_time = False, plots = False)
            
        else:    
            trajs = np.load(f'{file_name}')
        #trajs = np.unwrap(trajs)
        matrixes = []
        for t in time_select:
            m = matrix_at_t(trajs,t)
            matrixes.append(m)
        A=0
            
    if drift == True:
        drift = LangevinSimulator(dt=1e-4,torque=10)
        if load == False:
            t0 = time.time()
            trajs = drift.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts = int(npts), x0 = x0, Amplitude = 0, torque = 0,iteration = 0, save = False, print_time = False, plots = False)        
        else:

            trajs = np.load(f'{file_name}')
            
        trajs = np.unwrap(trajs)    
        matrixes = []
        for t in time_select:
            m = matrix_at_t(trajs,t)
            matrixes.append(m)    
        A=1
            
    if periodic == True:
        periodic = LangevinSimulator(dt=1e-4,torque=0)
        if load == False:
            trajs = periodic.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts = int(npts), x0 = x0, Amplitude = 1, torque = 0,iteration = 0, save = False, print_time = False, plots = False)
        else:
            trajs = np.load(f'{file_name}')
        matrixes = []
        for t in time_select:
            m = matrix_at_t(trajs,t)
            matrixes.append(m)
        A=2
            
    if tilted_periodic == True:
        tilted_periodic = LangevinSimulator(dt=1e-4,torque=10)
        if load == False:
            trajs = tilted_periodic.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts = int(npts), x0 = x0, Amplitude = 4, torque = 0,iteration = 0, save = False, print_time = False, plots = False)
        else:
            trajs = np.load(f'{file_name}')  
        matrixes = []
        for t in time_select:
            m = matrix_at_t(trajs,t)
            matrixes.append(m)
        A=3

    np.save(f'matrixes,repetitions,{A}',matrixes)

def plot_quadrant_hist(matrixes,time_select,repetitions,mu,D,torque,amplitude):

    def normal_theory(mu, t, x, D):
        """
        Theoretical probability density function for a free Brownian particle
        """
        # Calculate the standard deviation
        sigma = np.sqrt(2 * D * t)

        # Theoretical Gaussian distribution
        p_theory = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- ((x - mu)**2) / (2 * sigma**2))
        
        return p_theory     
    
    def defaveri_theory(mu, t, x, D):
        tiltedperiodic = LangevinSimulator(dt=1e-4,torque=torque)
        D_eff = tiltedperiodic.lifson_jackson_noforce(amplitude)
        """
        Theoretical probability density function for a Brownian particle in a potential.
        """
        # Calculate the standard deviation
        sigma = np.sqrt(2 * D * t)
        sigma_eff = np.sqrt(2 * D_eff * t)

        # Theoretical Gaussian distribution
        p_theory = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- ((x - mu)**2) / (2 * sigma_eff**2))*tiltedperiodic.integrand1(x,amplitude)
        
        return p_theory
          
    all_data = np.concatenate(matrixes)
    x_min, x_max = np.min(all_data), np.max(all_data)
    y_min, y_max = 0, max([np.histogram(m, bins=int(repetitions/2),density=True)[0].max() for m in matrixes])

    # Plotting histograms in quadrants
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    labels = [f't = {time_select[0] * 1e-4}s', f't = {time_select[1] * 1e-4}s', f't = {time_select[2] * 1e-4}s', f't = {math.ceil(time_select[3] * 1e-4)}s']
    axs = axs.flatten()

    for idx, m in enumerate(matrixes):
        t = time_select[idx]*1e-4
        axs[idx].hist(m, bins=int(repetitions/6), alpha=0.7,density=True)
        (mu, sigma) = norm.fit(m)
        xmin, xmax = axs[idx].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p_fit = norm.pdf(x, mu, sigma)
        if amplitude == 0:
            p_theory = normal_theory(mu,t,x,D)
        else :
            p_theory = defaveri_theory(mu,t,x,D)
        axs[idx].plot(x, p_fit, 'k', linewidth=2)
        axs[idx].plot(x, p_theory ,linestyle='--', color = 'r', linewidth=2)
        axs[idx].set_title(f'Time {labels[idx]}')
        axs[idx].set_xlim(x_min, x_max)
        axs[idx].set_ylim(y_min, y_max)
        axs[idx].set_xlabel('angle [rad]')
        axs[idx].set_ylabel('Occur.')
        axs[idx].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.2f}'.format(val)))

    plt.tight_layout()
    plt.show()

npts = 1e6
time_select = [1*int(npts/10),4*int(npts/10),7*int(npts/10),int(npts -1)] 
"""
produce_histograms(tilted_periodic=True, load=False, file_name='0ite_trajectories_1000000,nb_traj_1000points_amplitude_0kT,frequency_10_dt_0.0001_torque_0kT.npy')
produce_histograms(periodic=True, load=False, file_name='0ite_trajectories_1000000,nb_traj_1000points_amplitude_0kT,frequency_10_dt_0.0001_torque_0kT.npy') 
"""
def final_plot_hist_free():       
    matrixes_free = np.load('matrixes,repetitions,0.npy')
    free = LangevinSimulator(dt=1e-4)
    D = free.D
    plot_quadrant_hist(matrixes_free,time_select,1000,mu=np.pi,D=D,torque=0,amplitude=0)


def final_plot_hist_drift():       
    matrixes_drift = np.load('matrixes,repetitions,1.npy')
    drift = LangevinSimulator(dt=1e-4)
    D = drift.D
    plot_quadrant_hist(matrixes_drift,time_select,1000,mu=np.pi,D=D,torque=10,amplitude=0)


def final_plot_hist_periodic():       
    matrixes_periodic = np.load('matrixes,repetitions,2.npy')
    A = 1
    periodic = LangevinSimulator(dt=1e-4)
    D_eff = periodic.lifson_jackson_noforce(1)
    plot_quadrant_hist(matrixes_periodic,time_select,1000,mu=np.pi,D=D_eff,torque=0,amplitude=A)

def final_plot_hist_tiltedperiodic():       
    matrixes_tiltedperiodic = np.load('matrixes,repetitions,3.npy')
    A = 1
    tiltedperiodic = LangevinSimulator(dt=1e-4,torque=0)
    D = tiltedperiodic.D
    plot_quadrant_hist(matrixes_tiltedperiodic,time_select,1000,mu=np.pi,D=D,torque=0,amplitude=A)

final_plot_hist_tiltedperiodic()