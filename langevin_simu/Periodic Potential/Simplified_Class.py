# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:40:04 2024

@author: CBS
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:52:36 2024

Class particle v2

- Simulate large trajectories
- Compute MSD
- Compute mean MSD 
- Fitting data
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import time
from scipy.interpolate import interp1d
import scipy
from numdifftools import Derivative
from numba import njit

@njit
def _make_trace(x, npts, dUdx, dt, gamma, thermnoise):
    ''' part of the Langevin simulation, called by TiltedPeriodicDiffusion.make_trace() for numba acceleration '''
    for i in range(npts - 1):
        # idx of dUdx(x[i]):
        dUdx_idxi = int(np.round(x[i]*len(dUdx)/(2*np.pi)) - 1)
        force = -dUdx[dUdx_idxi]
        # Langevin eq:
        x[i+1] = x[i] + force*dt/gamma + thermnoise[i]
        x[i+1] = np.mod(x[i+1], 2*np.pi)
        #if np.isnan(x[i]): print(dUdx_idxi, force)
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
        self.load = 6 * np.pi * self.viscosity_NS_m2 * self.R_m 
        self.gamma = 8 * np.pi * self.viscosity_NS_m2 * self.R_m**3
        self.moment_inertia = (2/5)*self.m_kg*self.R_m**2
        self.tau = self.moment_inertia / self.gamma
        self.D = self.KT / self.gamma
        self.dt = dt # tau
        self.frequency = frequency
        self.space_step = 1e-8
        self.torque = torque*self.KT
        self.x0 = x0
        self.analytical = analytical
        self.x_pot = np.linspace(0, 2*np.pi, 50000)
    
    
    def make_potential_sin(self, ampl=None, plots=False):
        ''' periodic sinusoidal potential, ampl and tilt in KT units
            The barrier disappear at a tilt = ampl*pi*period (right?)
        '''
        U = ampl*self.KT*np.cos(self.x_pot*self.frequency) - self.torque*self.x_pot/(2*np.pi)*self.KT
        if plots:
            plt.figure('make_potential_sin', clear=True)
            plt.plot(self.x_pot, U/self.KT, ',')
            plt.ylabel('pot. [KT]')
        return U
    
    def main_traj_numba(self, N, A, U, x0, ide ):
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
      
    def run_parallel_numba(self, repetitions=None, n_jobs=5, npts = int(1e5), x0 = None, Amplitude = None, torque = 0,iteration = 0, save = True, print_time = False, plots = False):
        ''' parallel computations to generate multiple trajectories in one go
            parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
        '''
        # Parallel:
        if print_time == True:
            print('run_serial_parallel(): Parallel working...')
        t0 = time.time()
        sin_pot = self.make_potential_sin(ampl = Amplitude) 
        parallel_out = Parallel(n_jobs=n_jobs)(delayed(self.main_traj_numba)(N = npts,A = Amplitude, U = sin_pot, x0 = x0, ide=i) for i in range(repetitions))
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
            ax1.set_title(f'Ampl={(np.max(U)-np.min(U))/self.KT:.1f} KT(pkpk)    dt={self.dt:.1e} s    gamma={self.gamma:.1e} Nsm    D={self.D:.2f} m2/s', fontsize=8)
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
            ax4.set_title(f'mean speed: {(np.unwrap(x_wrap)[-1] - np.unwrap(x_wrap)[0])/t[-1]:.1f} Hz', fontsize=8)
            fig.tight_layout()
            print(f'make_trace(): plots done in {time.time() - t0} s.')
        return parallel_out
    
    def meta_parallel_numba(self, repetitions=None, n_jobs=5, npts = None, x0s = None, Amplitude = None, torque = 0, save = True): #x0 is an array of x0
        '''
        For very long trajectory, slice the generation in chunks 
        '''
        paquet = 1e8
        t0 = time.time()
        print('meta initialisation...')
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

    
    def numba_msd(self, traj, time_end=1/4, msd_nbpt = 2000,print_time=False):
        traj = np.unwrap(traj)
        t0 = time.time()
        max_lagtime = int(len(traj) * time_end)
        total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])
        msd_results = single_msd(traj, total_lag_time)
        if print_time==True:
            print(f'MSD_numba done in {time.time() - t0:.1f} s')
        
        return msd_results 
        
    
    def full_auto_trajs_mean_msd(self,ampl_range = np.arange(8),nb_traj = 30, npts = int(1e8) ):
        """
        Generate trajectories, calculate and mean MSD for each value of amplitude.
        Parameters
        ----------
        ampl_range : array, optional
            Range of amplitude to iterate over. The default is np.arange(8).
        nb_traj : TYPE
            Number of trajectories generated per iteration.
        npts : TYPE, optional
            length of a trajectory. The default is None.

        Returns
        -------
        None.

        """
        trajectories_box = []
        mean_msd_box = []
        x0s = np.zeros(npts)
        for A in ampl_range:
            trajs = self.meta_parallel_numba(self, repetitions=None, n_jobs=5, npts = None, x0s = None, Amplitude = A, torque = 0)
            trajectories_box.append(trajs)
            time_axis,mean_msd = self.mean_msd_and_time_numba(trajs, time_end=1/4, msd_nbpt = 2000, nb_traj = nb_traj)
            mean_msd_box.append(mean_msd)
        
        """
        ipython lines :
            D = DiffusionSimulation2(dt=1e-4)
            time_axis,mean_msd_box,trajectories_box = D.full_auto_trajs_mean_msd(npts=int(1e6))
        """
        return time_axis,mean_msd_box,trajectories_box
    
    
    
    """
    Lifson and Jackson methods
    """

    def integrand1(self, x, amplitude):
        return np.exp(self.tilted_periodic_potential(amplitude, x))
    
    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / self.frequency, np.pi / self.frequency, args=(amplitude))
        return result
    
    def lifson_jackson_noforce(self, amplitude): #Meet einstein coeff at 0 barrier
        lifson_jackson1 = self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))
        return lifson_jackson1 
        





