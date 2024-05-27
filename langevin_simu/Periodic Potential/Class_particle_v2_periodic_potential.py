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

class DiffusionSimulation2:
    
                  # periodic potential spatial variable 


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
        
    def tilted_periodic_potential(self, A, x):
        return self.torque*x + A * np.sin(x * self.frequency)
    
    def potential_analytic_derivative(self,A,x):
        return self.torque + A * self.frequency*np.cos(x * self.frequency)
    
    def potential_derivative(self, A, x, dx=1e-6):
        # Compute the derivative numerically using finite differences
        derivative_at_x = (self.tilted_periodic_potential(A, x + dx) - self.tilted_periodic_potential(A, x - dx)) / (2 * dx)
        return derivative_at_x
    
    def box_muller(self,N):
        u1 = np.random.rand(N)
        u2 = np.random.rand(N)
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return z

    def generate_seq(self, N):
        W_seq = self.box_muller(N)
        return W_seq
    
    
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
    
    def main_traj_method(self, N, A):
        """ Perform the overdamped rotational Langevin dynamic simulation in a given potential, all units are in S.I.

        Args:
            N (int): trajectory length
            A (float): barrier amplitude 

        Returns:
            array: angular trajectory
        """
        
        w = self.generate_seq(N)
        A *= self.KT
        x0 = self.x0
        x = x0
        if self.analytical == False : 
            positions = [ x := (x - (1 / self.gamma) * self.dt * self.potential_derivative(A,x) + np.sqrt(2 * self.D * self.dt) * w[i]) for i in range(N)]
        else :
            positions = [ x := (x - (1 / self.gamma) * self.dt * self.potential_analytic_derivative(A,x) + np.sqrt(2 * self.D * self.dt) * w[i]) for i in range(N)]

        np.insert(positions,[0],x0)
        positions = np.array(positions) 
        
        positions.astype(np.float32)
        return positions
    
    def main_traj_debug(self,N,A):
        """ Perform the overdamped rotational Langevin dynamic simulation in a given potential, all units are in S.I.
            allow to decompose the trajectory in a stochastic and deterministic part.

        Args:
            N (int): trajectory length
            A (float): barrier amplitude 

        Returns:
            array: angular trajectory, thermal_noise, deterministic
        """
        w = self.generate_seq(N)
        A *= self.KT
        x0 = 0
        x = self.x0
        positions = []
        deter = []
        stocha = []
        for i in range(N):
            positions.append(x)    
            if self.analytical == False : 
                deterministic = (1 / self.gamma) * self.dt *self.potential_derivative( A, x)
            if self.analytical == True :
                deterministic = (1 / self.gamma) * self.dt *self.potential_analytic_derivative( A, x)
            stochastic = np.sqrt(2 * self.D * self.dt) * w[i] 
            x += -deterministic + stochastic
            deter.append(deterministic)
            stocha.append(stochastic)
                
        return positions,stocha,deter
    
    def regular_linear_msd(self, traj = None, time_end = 1/4, time_skip = 100):
        """Compute the mean square displacement by iterating through an array of lag times

        Args:
            traj (array): The targeted array of positions.
            time_end (float): The iterated fraction of the trajectory length.
            time_skip (int): Quantity of points skiped when performing msd.

        Returns:
            array: The mean square displacement
            array: The lag times array    
        """
        lag_time = np.arange(0, int(len(traj) * time_end), time_skip)
        msd = []
        for j in lag_time:
            if j == 0 :
                msd.append(0)
                continue
            msd.append(np.mean((traj[:-j] - traj[j:])**2))
        return np.array(msd)  
    
    
    def run_parallel_numba(self, repetitions=None, n_jobs=5, npts = int(1e5), x0 = None, Amplitude = None, torque = 0,iteration = 0, save = True, print_time = False):
        ''' parallel computations of generate_traj() 
            parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
        '''
        # Parallel:
        if print_time == True:
            print(f'run_serial_parallel(): Parallel working...')
        t0 = time.time()
        sin_pot = self.make_potential_sin(ampl = Amplitude) 
        parallel_out = Parallel(n_jobs=n_jobs)(delayed(self.main_traj_numba)(N = npts,A = Amplitude, U = sin_pot, x0 = x0, ide=i) for i in range(repetitions))
        if print_time == True:
            print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
        if save == True:
            np.save(f'{iteration}ite_trajectories_{npts:.0f},nb_traj_{repetitions}points_amplitude_{Amplitude}kT,frequency_{self.frequency}_dt_{self.dt}_torque_{torque:.0f}kT',parallel_out)
        return parallel_out
    
    def meta_parallel_numba(self, repetitions=None, n_jobs=5, npts = None, x0s = None, Amplitude = None, torque = 0, save = True): #x0 is an array of x0
        '''
        Because trajectory length >= 1e6points induces segmentation error, take the last trajectory point and inject it as the x0 of a new trajectory
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

          
            
    def run_parallel(self, repetitions=None, n_jobs=5, npts = None, Amplitude = 0, torque = 0):
        ''' parallel computations of generate_traj() 
            parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
        '''
        # Parallel:
        print(f'run_serial_parallel(): Parallel working...')
        t0 = time.time()
        parallel_out = Parallel(n_jobs=n_jobs)(delayed(self.main_traj_method)(N = npts,A = Amplitude) for i in range(repetitions))
        print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
        np.save(f'trajectories_{npts:.0f},nb_traj_{repetitions}points_amplitude_{Amplitude}kT,frequency_{self.frequency}_dt_{self.dt}_torque_{torque:.0f}kT',parallel_out)
        return parallel_out
    
    def run_parallel_debug(self, repetitions=None, n_jobs=5, npts = None, Amplitude = 0, torque = 0):
        ''' parallel computations of generate_traj() 
            parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
        '''
        # Parallel:
        print(f'run_serial_parallel(): Parallel working...')
        t0 = time.time()
        parallel_out = Parallel(n_jobs=n_jobs)(delayed(self.main_traj_debug)(N = npts,A = Amplitude) for i in range(repetitions))
        print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
        np.save(f'trajectories_{npts:.0f},nb_traj_{repetitions}_points_amplitude_{Amplitude}kT,frequency_{self.frequency}_dt_{self.dt}_torque_{torque:.0f}kT',parallel_out)
        return parallel_out
    
    
    def parallel_no_chunk(self, traj, time_end=1/4, msd_nbpt = 2000,n_jobs=5):
        """

        Parameters
        ----------
        traj : array
            trajectory of angular displacement.
        time_end : float, optional
            fraction of the trajectory array on which is performed the msd . The default is 1/4.
        msd_nbpt : int, optional
            Number of MSD points. The default is 2000.
        n_jobs : TYPE, optional
            number of worker assigned to the task. The default is 5.

        Returns
        -------
        array
            MSD array.

        """
        t0 = time.time()
        max_lagtime = int(len(traj) * time_end)
        total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])
        
        
        def single_msd(lag):
            return np.mean((traj[:-lag] - traj[lag:]) ** 2)
        
        msd_results = Parallel(n_jobs=n_jobs)(
            delayed(single_msd)(lag) for lag in total_lag_time 
        )
        print(f'mean_msd_no_chunk(): Parallel done in {time.time() - t0:.1f} s')
        return msd_results
    
    def numba_msd(self, traj, time_end=1/4, msd_nbpt = 2000,print_time=False):
        t0 = time.time()
        max_lagtime = int(len(traj) * time_end)
        total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])
        msd_results = single_msd(traj, total_lag_time)
        if print_time==True:
            print(f'MSD_numba done in {time.time() - t0:.1f} s')
        
        return msd_results 
        
    def load_traj_and_logmsd_chunk(self, traj_name = None,nb_chunks=10, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = None):
        trajs = np.load(f'{traj_name}.npy')
        for i in range(len(trajs[:nb_traj])):
            self.run_parallel_msd_chunk_log(nb_chunks=nb_chunks, n_jobs=n_jobs, time_end=time_end, msd_nbpt = msd_nbpt, traj=trajs[i],n= f'logmsd{traj_name},num_{i:.0f}')

    def mean_msd_and_time_axis_std(self, trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = None):
        t0 = time.time()
        msd_matrix = []
        max_lagtime = int(len(trajs[0]) * time_end)
        for i in range(len(trajs[:nb_traj])):
            msd_matrix.append(self.parallel_no_chunk(trajs[i], time_end=1/4, msd_nbpt = 2000,n_jobs=5))
        mean_msd = np.concatenate(([0],np.mean(msd_matrix, axis=0)))
        std = np.concatenate(([0],np.std(msd_matrix,axis=0)))
        time_axis = np.concatenate(([0],np.unique((np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt)))))) 
        print(f'mean_msd_no_chunk(): Parallel done in {time.time() - t0:.1f} s')
        return time_axis,mean_msd,std,msd_matrix
    

    def mean_msd_and_time_axis(self, trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = None):
        t0 = time.time()
        msd_matrix = []
        max_lagtime = int(len(trajs[0]) * time_end)
        for i in range(len(trajs[:nb_traj])):
            msd_matrix.append(self.parallel_no_chunk(trajs[i], time_end=1/4, msd_nbpt = 2000,n_jobs=5))
        mean_msd = np.concatenate(([0],np.mean(msd_matrix, axis=0)))
        time_axis = np.concatenate(([0],np.unique((np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt)))))) 
        print(f'mean_msd_no_chunk(): Parallel done in {time.time() - t0:.1f} s')
        return time_axis,mean_msd
    
    def mean_msd_and_time_numba(self, trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = None):
        t0 = time.time()
        msd_matrix = []
        max_lagtime = int(len(trajs[0]) * time_end)
        for i in range(len(trajs[:nb_traj])):
            msd_matrix.append(self.numba_msd(trajs[i], time_end=1/4, msd_nbpt = 2000))
        mean_msd = np.concatenate(([0],np.mean(msd_matrix, axis=0)))
        time_axis = np.concatenate(([0],np.unique((np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt)))))) 
        print(f'mean_msd_no_chunk(): Parallel done in {time.time() - t0:.1f} s')
        return time_axis,mean_msd
    
    def mean_msd_and_time_axis_debug(self, trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = None):
        t0 = time.time()
        msd_matrix = []
        max_lagtime = int(len(trajs[0][0]) * time_end)
        for i in range(len(trajs[:nb_traj])):
            msd_matrix.append(self.parallel_no_chunk(trajs[i][0], time_end=1/4, msd_nbpt = 2000,n_jobs=5))
        mean_msd = np.concatenate(([0],np.mean(msd_matrix, axis=0)))
        time_axis = np.concatenate(([0],np.unique((np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt)))))) 
        print(f'mean_msd_no_chunk(): Parallel done in {time.time() - t0:.1f} s')
        return time_axis,mean_msd
    
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
        
    def lifson_jackson_force(self, amplitude):
        lifson_jackson2 = (self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))) * (np.sinh(self.torque*(2*np.pi / self.frequency)/2))/(self.torque*(2*np.pi / self.frequency)/2)
        return lifson_jackson2

    
def store_msd_given_Amplitude(Ampli):
    
    traj4 = np.load(f'trajectories_10000000points_amplitude_{Ampli}kT_dt_0.0001_torque_0kT.npy')
    traj45 = np.load(f'trajectories_2000000points_amplitude_{Ampli}kT_dt_0.0005_torque_0kT.npy')
    traj3 = np.load(f'trajectories_1000000points_amplitude_{Ampli}kT_dt_0.001_torque_0kT.npy')
    traj35 = np.load(f'trajectories_200000points_amplitude_{Ampli}kT_dt_0.005_torque_0kT.npy')
    traj2 = np.load(f'trajectories_100000points_amplitude_{Ampli}kT_dt_0.01_torque_0kT.npy')
    traj25 = np.load(f'trajectories_20000points_amplitude_{Ampli}kT_dt_0.05_torque_0kT.npy')
    traj_list = [traj4,traj45,traj3,traj35,traj2,traj25]
    dt_list = [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2]
    counter = 0
    for trajs in traj_list:
        p = DiffusionSimulation2(dt = dt_list[counter])
        time_axis,mean_msd,std,matrix = p.mean_msd_and_time_axis(trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = 50)
        np.save(f't,msd,dt={p.dt_s},{Ampli}kT',[time_axis*dt_list[counter],mean_msd])
        counter += 1
        
        
        
def qth_moment(q,t,D_eff):
    t *= (0.1648)/(2*np.pi/26)**2
    return((2**q)/(np.sqrt(np.pi)))*(D_eff*t)**(q/2)*scipy.special.gamma((1+q)/2)
    
def second_moment(t,D_eff):
    return(4/(np.sqrt(np.pi)))*(D_eff*t)*scipy.special.gamma(3/2)

def generate_theory_msd(absciss,D_fit):
    theoretical_msd = []
    for t in absciss:
        msd_theo = qth_moment(2,t,D_fit)
        theoretical_msd.append(msd_theo)
    return theoretical_msd

def asdefaveri_msd(A,dt):
    p = DiffusionSimulation2(dt=dt)
    a = 2*np.pi/p.frequency
    D = p.D           
    D_eff = p.lifson_jackson_noforce(A)
    
    def defaveri_stationary(A):
        X = np.linspace(0,a,1000)
        Y = boltz_exp(A,X)
        Z = np.mean(Y)
        return (np.mean((X**2)*Y)/Z)   
     
    def boltz_exp(A,x):
        return np.exp(-p.tilted_periodic_potential(A, x))
        
    def linear_D(t,D,shift):
        return (2*D*t + shift)/(a*a)
    
    t, msd = np.load(f't,msd,dt={dt},{A}kT.npy')
    absciss = 2*D*t
    norm_msd = msd/(a*a)
    
    min_absciss_value = 1  # Adjust this value as needed
    
    window_indices = np.where(absciss > min_absciss_value)[0]
    absciss_window = absciss[window_indices]
    norm_msd_window = norm_msd[window_indices]
    
    popt, pcov = scipy.optimize.curve_fit(linear_D, absciss_window, norm_msd_window,)
    D_fit, shift = popt[0], popt[1]
    print('D_fit = ',D_fit*2*D)
    print('D_eff = ',D_eff)
    stationary_state = defaveri_stationary(A) 
    return absciss,norm_msd,D_fit,D_eff,shift,stationary_state,a


