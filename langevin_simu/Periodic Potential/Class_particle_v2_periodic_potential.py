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

class DiffusionSimulation2:
    def __init__(self,frequency = 26, torque = 0, dt = None):
        # Constants
        self.T_K = 300
        self.k_b = 1.3806452e-23
        self.R_m = 1e-6
        self.m_kg = 1.1e-14 
        self.viscosity_NS_m2 = 0.001
        self.load = 6 * np.pi * self.viscosity_NS_m2 * self.R_m 
        self.rotational_drag = 8 * np.pi * self.viscosity_NS_m2 * self.R_m**3
        self.moment_inertia = (2/5)*self.m_kg*self.R_m**2
        self.tau = self.moment_inertia / self.rotational_drag
        self.einstein_diffusion = self.k_b * self.T_K / self.load
        self.rotational_einstein_diff = self.k_b * self.T_K / self.rotational_drag
        self.dt_s = dt # tau
        self.frequency = frequency
        self.space_step = 1e-12
        self.torque = torque*self.T_K*self.k_b
        
    def tilted_periodic_potential(self, A, x):
        return self.torque*x + A * np.cos(x * self.frequency)
        
    def generate_seq(self, N):
        W_seq = np.random.normal(0, 1, N)
        return W_seq

    def main_traj(self, N, A):
        """ Perform the overdamped rotational Langevin dynamic simulation in a given potential, all units are in S.I.

        Args:
            N (int): trajectory length
            A (float): barrier amplitude 

        Returns:
            array: angular trajectory
        """
        w = self.generate_seq(N)
        A *= self.k_b * self.T_K
        x = 0
        positions = [ x := (x - (1 / self.rotational_drag) * self.dt_s *((self.tilted_periodic_potential(A, x + self.space_step) - self.tilted_periodic_potential(A, x - self.space_step)) / (2*self.space_step)) + np.sqrt(2 * self.rotational_einstein_diff * self.dt_s) * w[i]) for i in range(N)]
        np.insert(positions,[0],0)
        positions = np.array(positions) 
        
        positions.astype(np.float32)
        return positions
    
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
    
    
    def run_parallel(self, repetitions=None, n_jobs=5, npts = None, Amplitude = 0, torque = 0):
        ''' parallel computations of generate_traj() 
            parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
        '''
        # Parallel:
        print(f'run_serial_parallel(): Parallel working...')
        t0 = time.time()
        parallel_out = Parallel(n_jobs=n_jobs)(delayed(self.main_traj)(N = npts,A = Amplitude) for i in range(repetitions))
        print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
        np.save(f'trajectories_{npts:.0f},nb_traj_{repetitions}points_amplitude_{Amplitude}kT,frequency_{self.frequency}_dt_{self.dt_s}_torque_{torque:.0f}kT',parallel_out)
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
        max_lagtime = int(len(traj) * time_end)
        total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])
        
        
        def single_msd(lag):
            return np.mean((traj[:-lag] - traj[lag:]) ** 2)
        
        msd_results = Parallel(n_jobs=n_jobs)(
            delayed(single_msd)(lag) for lag in total_lag_time 
        )
        
        return msd_results
        
    def load_traj_and_logmsd_chunk(self, traj_name = None,nb_chunks=10, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = None):
        trajs = np.load(f'{traj_name}.npy')
        for i in range(len(trajs[:nb_traj])):
            self.run_parallel_msd_chunk_log(nb_chunks=nb_chunks, n_jobs=n_jobs, time_end=time_end, msd_nbpt = msd_nbpt, traj=trajs[i],n= f'logmsd{traj_name},num_{i:.0f}')

    def mean_msd_and_time_axis(self, trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = None):
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
    

    """
    Lifson and Jackson methods
    """

    def integrand1(self, x, amplitude):
        return np.exp(self.tilted_periodic_potential(amplitude, x))
    
    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / self.frequency, np.pi / self.frequency, args=(amplitude))
        return result
    
    def lifson_jackson_noforce(self, amplitude): #Meet einstein coeff at 0 barrier
        lifson_jackson1 = self.rotational_einstein_diff * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))
        return lifson_jackson1 
        
        
        
def compare_ampli(ampli_min,ampli_max,dt):
    total_time = int(1e8*1e-5)
    for A in range(ampli_min,ampli_max):
        p = DiffusionSimulation2(dt=dt)
        p.run_parallel( repetitions=50, n_jobs=5, npts = nb_points, Amplitude = Ampli, torque = 0)

    
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
    D = p.rotational_einstein_diff           
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

