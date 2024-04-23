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



class DiffusionSimulation2:
    def __init__(self,frequency = 26, torque = 0, dt=1e-5):
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
    
    
    def run_parallel(self, repetitions=10, n_jobs=5, npts = None, Amplitude = 0, torque = 0):
        ''' parallel computations of generate_traj() 
            parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
        '''
        # Parallel:
        print(f'run_serial_parallel(): Parallel working...')
        t0 = time.time()
        parallel_out = Parallel(n_jobs=n_jobs)(delayed(self.main_traj)(N = npts,A = Amplitude) for i in range(repetitions))
        print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
        np.save(f'trajectories_{npts:.0f}points_amplitude_{Amplitude:.0f}kT_dt_{self.dt_s}_torque_{torque:.0f}kT',parallel_out)
        return parallel_out
    
    def calculate_msd_chunk_log(self, current_chunk, nb_chunks=10, time_end=1/4, traj= None, msd_nbpt = None):
        traj = np.array(traj)
        msd_chunk = [np.mean((traj[:-lag] - traj[lag:]) ** 2) for lag in current_chunk]
        return msd_chunk
    
    def run_parallel_msd_chunk_log(self, nb_chunks=10, n_jobs=5, time_end=1/4, msd_nbpt = 2000, traj=None,n=1234):
        
        max_lagtime = int(len(traj) * time_end)
        total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])
        
        def find_div_nb_chunk(total_lag_time):
            count = len(total_lag_time)
            while count%nb_chunks != 0:
                count = count - 1
            return count
        
        greatest_div = find_div_nb_chunk(total_lag_time)
        chunks_size = int((greatest_div) / nb_chunks)
        chunk_list = []
         
      
        for j in range(nb_chunks+1):
            chunk_list.append(total_lag_time[int(j*chunks_size):int((j+1)*chunks_size)])
            if j == nb_chunks:
                chunk_list.append(total_lag_time[int(j*chunks_size):-len(total_lag_time)%nb_chunks])
        
        msd_results = Parallel(n_jobs=n_jobs)(delayed(self.calculate_msd_chunk_log)(
            current_chunk, nb_chunks=nb_chunks, time_end=time_end, traj=traj, msd_nbpt=None
        ) for current_chunk in chunk_list)
        
        
        final_msd = np.concatenate(msd_results)
        np.save(f'{n}', final_msd)
        return final_msd
    
    def parallel_no_chunk(self, traj, time_end=1/4, msd_nbpt = 2000,n_jobs=5):
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

def plot_drift_msd_std():
    traj_drift = np.load('trajectories_100000points_amplitude_0kT_dt_1e-05_torque_20kT.npy')
    p = DiffusionSimulation2()
    time_d,mean_msd_d,std,msd_matrix_d = p.mean_msd_and_time_axis(traj_drift,nb_chunks=10, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = 10)
    time_d *= 1e-5
    plt.plot(time_d,mean_msd_d, label = 'Numerical MSD')
    plt.fill_between(time_d, mean_msd_d -std, mean_msd_d + std, color='gray', alpha=0.6, label='std')
    plt.xlabel('Lag time (s)')
    plt.ylabel('Square displacement (rad²)')
    plt.show()


def compare_dt_10kT():
    total_time = int(1e8*1e-5)
    for dt in [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2]:
        p = DiffusionSimulation2(dt)
        nb_points = int(total_time/dt)
        p.run_parallel( repetitions=10, n_jobs=5, npts = nb_points, Amplitude = 10, torque = 0)

def store_msd():
    
    traj4 = np.load('trajectories_10000000points_amplitude_10kT_dt_1e-05_torque_0kT.npy')
    traj45 = np.load('trajectories_2000000points_amplitude_10kT_dt_1e-05_torque_0kT.npy')
    traj3 = np.load('trajectories_1000000points_amplitude_10kT_dt_1e-05_torque_0kT.npy')
    traj35 = np.load('trajectories_200000points_amplitude_10kT_dt_1e-05_torque_0kT.npy')
    traj2 = np.load('trajectories_100000points_amplitude_10kT_dt_1e-05_torque_0kT.npy')
    traj25 = np.load('trajectories_20000points_amplitude_10kT_dt_1e-05_torque_0kT.npy')
    traj_list = [traj4,traj45,traj3,traj35,traj2,traj25]
    dt_list = [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2]
    counter = 0
    for trajs in traj_list:
        p = DiffusionSimulation2(dt = dt_list[counter])
        counter += 1
        time_axis,mean_msd,std,msd_matrix = p.mean_msd_and_time_axis(trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = 10)
        np.save(f't,msd,std,dt={p.dt_s},10kT',[time_axis,mean_msd,std])
    
def plot_msd():
    dt_list = [0.0001,0.0005,0.001,0.005,0.01,0.05]
    
    for dt in dt_list:
        t,msd,_ = np.load(f't,msd,std,dt={dt},10kT.npy')
        t=t*dt
        plt.loglog(t,msd,label=f'{dt}')
        plt.legend()
        



