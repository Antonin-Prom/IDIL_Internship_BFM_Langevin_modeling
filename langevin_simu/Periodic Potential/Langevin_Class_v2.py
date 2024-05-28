# -*- coding: utf-8 -*-

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
from matplotlib.lines import Line2D


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
      
    def brutal_msd(self,repetition=None,N=None,Amplitude=None,x0=[0],ide=0,msd_nbpt = 500, time_end=1/4,save=True):
        t0 = time.time()
        msd_box = []
        U = self.make_potential_sin(ampl = Amplitude) 
        for i in range(repetition):
            traj = self.main_traj_( N, Amplitude, U, x0, ide )
            msd_box.append(self.msd(traj, time_end=time_end, msd_nbpt = msd_nbpt, print_time=False))
            print(f'MSD done in {time.time() - t0:.1f} s')
            del traj
        max_lagtime = int(N * time_end)
        mean_msd = np.concatenate(([0],np.mean(msd_box, axis=0)))
        time_axis = np.concatenate(([0],np.unique((np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt)))))) 
        if save == True:
            np.save(f't,msd_{N}npts_{repetition}rep_torque_{self.torque}kT_dt={self.dt},cylindric',[time_axis,mean_msd])
        return time_axis,mean_msd
    
    def brutal_msd_amplitude_range(self,ampl_range=[0,5,10,20],repetition=None,N=None,x0=[0],ide=0,msd_nbpt = 500, time_end=1/4,save=True):
        time_box,mean_msd_box = [],[]
        for A in ampl_range:
            time_axis,mean_msd = self.brutal_msd(repetition=repetition,N=N,Amplitude=A,x0=x0,ide=ide,msd_nbpt = msd_nbpt, time_end=time_end,save=save)
            time_box.append(time_axis)
            mean_msd_box.append(mean_msd)
        np.save(f'A=0,5,10,20,t,msd_{N}npts_{repetition}rep_torque_{self.torque}kT_dt={self.dt},cylindric',[time_box,mean_msd_box])
        return time_box,mean_msd_box
    
    def run_parallel_numba(self, repetitions=None, n_jobs=5, npts = int(1e5), x0 = None, Amplitude = None, torque = 0,iteration = 0, save = False, print_time = False, plots = False):
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
    
    def msd(self, traj, time_end=1/4, msd_nbpt = 500,print_time=False):
        traj = np.unwrap(traj)
        t0 = time.time()
        max_lagtime = int(len(traj) * time_end)
        total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])
        msd_results = single_msd(traj, total_lag_time)
        if print_time==True:
            print(f'MSD_numba done in {time.time() - t0:.1f} s')
        
        return msd_results 
        
    def mean_msd_and_time(self, trajs, n_jobs=5, time_end=1/4, msd_nbpt = 500, nb_traj = None,t = int(1e6)):
        """
        Compute the msd for a set of trajectories, than mean the msds.

        Returns
        -------
        time_axis : array
            time points not in second
        mean_msd : array
        """
        t0 = time.time()
        msd_matrix = []
        max_lagtime = int(len(trajs[0]) * time_end)
        for i in range(len(trajs[:nb_traj])):
            msd_matrix.append(self.msd(trajs[i], time_end=1/4, msd_nbpt = msd_nbpt))
        mean_msd = np.concatenate(([0],np.mean(msd_matrix, axis=0)))
        time_axis = np.concatenate(([0],np.unique((np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt)))))) 
        print(f'mean_msd_no_chunk(): Parallel done in {time.time() - t0:.1f} s')
        
        
        """
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
            """
        return time_axis,mean_msd
    
    def full_auto_trajs_mean_msd(self,ampl_range = np.arange(8),nb_traj = 30, npts = int(1e7),msd_nbpt=500, load = False, trajs_name = None ):
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
        time_box = []
        x0s = np.zeros(nb_traj)
        for A in ampl_range:
            if load == True:
                trajs = np.load(f'{trajs_name}.npy')
            else:               
                trajs = self.chunkify_traj(repetitions=nb_traj, n_jobs=5, npts = npts, x0s = x0s, Amplitude = A, torque = 0,save=False)
                trajectories_box.append(trajs)
            time_axis,mean_msd = self.mean_msd_and_time(trajs, time_end=1/4, msd_nbpt = msd_nbpt, nb_traj = nb_traj)
            mean_msd_box.append(mean_msd)
            time_axis *= self.dt
            time_box.append(time_axis)
        """
        ipython lines :
            D = DiffusionSimulation2(dt=1e-4)
            time_axis,mean_msd_box,trajectories_box = D.full_auto_trajs_mean_msd(npts=int(1e6))
        """
        return time_box,mean_msd_box,trajectories_box
    
    
    """
    Lifson and Jackson methods
    """

    def integrand1(self, x, amplitude):
        return np.exp(amplitude*np.cos(x*self.frequency))
    
    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / self.frequency, np.pi / self.frequency, args=(amplitude))
        return result
    
    def lifson_jackson(self, amplitude): #Meet einstein coeff at 0 barrier
        if self.torque == 0:
            lifson_jackson1 = self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))
            return lifson_jackson1
        else:
            F = self.torque/(2*np.pi) #KT already taken in acount
            lifson_jackson2 = (self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))) * ((np.sinh(F*(2*np.pi / self.frequency)/2))/(F*(2*np.pi / self.frequency)/2))**2
            return lifson_jackson2            
 
    
    
    def normalised_loglog_msd_fit(self,ampl_range,npts):
        
        time_box,mean_msd_box,trajectories_box = self.full_auto_trajs_mean_msd(ampl_range = ampl_range,nb_traj = 20, npts = npts )
        time_box,mean_msd_box = np.asarray(time_box, dtype=np.float32),np.asarray(mean_msd_box, dtype=np.float32)
        np.save(f'To_plot_t,{npts}npts_msd_torque_{self.torque}kT_dt={self.dt},cylindric',[time_box,mean_msd_box])
        
    def fit_and_plot_msd(self, time_axis_boxbox, mean_msd_boxbox, ampl_range, torque_range):
        """
        boxbox for different torques
        box for different amplitudes
    
        Parameters
        ----------
        time_axis_boxbox : list of numpy arrays
            List containing arrays of time axes for different torques.
        mean_msd_boxbox : list of numpy arrays
            List containing arrays of mean MSDs for different torques.
        ampl_range : list
            List of amplitude values.
        torque_range : list
            List of torque values.
    
        Returns
        -------
        None
        """
        
        D = self.D    
        a = 2 * np.pi / self.frequency
        
        def defaveri_stationary(A):
            X = np.linspace(0, a, 1000)
            Y = boltz_exp(A, X)
            Z = np.mean(Y)
            return (np.mean((X ** 2) * Y) / Z)   
        
        def boltz_exp(A, x):
            return np.exp(self.analytical_potential(x, A))
            
        def linear_D(t, D):
            return (2 * D * t)
        
        def parabolic_msd(t,D,v_eff):
            return (2 * D * t) + (v_eff**2)*t**2 
        
        results = []
        
        # Define color and marker lists
        colors = plt.cm.viridis(np.linspace(0, 1, len(ampl_range)))
        markers = ['o', 'x','v', 's', '^', 'D',  'p', '*', '+', 'h'][:len(torque_range)]
        
        plt.style.use('seaborn-v0_8-colorblind')
        
        for j in range(len(torque_range)):
            time_axis_box = time_axis_boxbox[j]
            mean_msd_box = mean_msd_boxbox[j]
            self.torque = torque_range[j]
            F = self.KT*self.torque/(2*np.pi)
            v_eff = F/self.gamma
            print('JJJJJJJ,V_eff',j,v_eff)
            for i, A in enumerate(ampl_range):
                U = self.make_potential_sin(A)
                absciss = time_axis_box[i] * self.D / (a * a)
                norm_msd = mean_msd_box[i]

                min_absciss_value = 0
                window_indices = np.where(absciss > min_absciss_value)[0]

                D_eff = self.lifson_jackson(A)

                popt, _ = scipy.optimize.curve_fit(parabolic_msd, time_axis_box[i][window_indices], mean_msd_box[i][window_indices])
                D_fit = popt[0]

                print(f'For A = {A} and torque = {self.torque}kT, D_fit = {D_fit}')
                print(f'D_eff = {D_eff}')

                frac = 1.5
                if A != 0:
                    plt.plot(time_axis_box[i][int(len(absciss)/frac):], (parabolic_msd(time_axis_box[i], D_eff, v_eff)/A**2)[int(len(absciss)/frac):], linewidth=1, color=colors[i])
                else:
                    plt.plot(time_axis_box[i][int(len(absciss)/frac):], (parabolic_msd(time_axis_box[i], D_eff, v_eff)[int(len(absciss)/frac):]), linewidth=1, color=colors[i])

                num_points = 20
                indices = np.linspace(0, len(norm_msd) - 1, num_points).astype(int)
                plt.scatter(time_axis_box[i][indices], norm_msd[indices], color=colors[i], marker=markers[j], s=20)
        #plt.scatter(time_axis_box[i][indices], norm_msd[indices], color=colors[i], marker=markers[j], s=20)
    
        plt.xlabel(r'${Dt}/{a^2}$', fontsize=16)
        plt.ylabel(r'${\langle \theta^2 \rangle}/{a^2}$', fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(0.00001, 1000000)
        # Custom legend for amplitudes (colors)
        
        legend_amplitudes = [Line2D([0], [0], color=colors[i], lw=4, label=f'A={ampl_range[i]:.1f} KT') for i in range(len(ampl_range))]
        
        # Custom legend for torques (markers)
        legend_torques = [Line2D([0], [0], color='black', marker=markers[j], linestyle='None', markersize=10, label=f'Torque={torque_range[j]}') for j in range(len(torque_range))]
        
        # Combine the legends
        plt.legend(handles=legend_amplitudes + legend_torques, fontsize=8)

        #plt.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        plt.title('dt=10^-4, traj_length 10^7pt, msd 500pt')
        # Save the figure
        plt.savefig('msd_plot_rod.png', dpi=300)
        plt.show()
 
ampl_range = [0,0.5,1,1.5,2]    

"""
tor = 0
J = LangevinSimulator(dt=1e-4, torque = tor)
J.normalised_loglog_msd_fit(ampl_range,int(1e8))
tor = 10
J = LangevinSimulator(dt=1e-4, torque = tor)
J.normalised_loglog_msd_fit(ampl_range,int(1e8))
tor = 20
J = LangevinSimulator(dt=1e-4, torque = tor)
J.normalised_loglog_msd_fit(ampl_range,int(1e8))

"""

J = LangevinSimulator(dt=1e-4, torque = 0)


t0,msd0 = np.load('To_plot_t,10000000npts_msd_torque_0kT_dt=0.0001,cylindric.npy')
t10,msd10 = np.load('To_plot_t,10000000npts_msd_torque_10kT_dt=0.0001,cylindric.npy')
t20,msd20 = np.load('To_plot_t,10000000npts_msd_torque_20kT_dt=0.0001,cylindric.npy')

t_boxbox = [t0,t10,t20]
msd_boxbox = [msd0,msd10,msd20]
torque_range=[0,10,20]
J.fit_and_plot_msd(t_boxbox,msd_boxbox,ampl_range,torque_range)








