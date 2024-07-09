"""
Simplified version of the Dynamic Langevin Class.

    ### TO DO ###
    
    
    
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
from matplotlib.lines import Line2D
from tqdm import tqdm
import seaborn as sns



@njit
def _make_trace(x, npts, dUdx, dt, gamma, thermnoise):
    ''' langevin finite difference integrator, called by main_traj_ for numba acceleration '''
    for i in range(npts - 1):
        dUdx_idxi = int(np.round(x[i]*len(dUdx)/(2*np.pi)) - 1)
        force = -dUdx[dUdx_idxi]
        # Langevin eq:
        x[i+1] = x[i] + force*dt/gamma + thermnoise[i]
    return x


@njit(parallel=True)
def single_msd(traj, total_lag_time):
    ''' Calculate the MSD over a single traj, for a given set of lag times'''
    msd_results = np.zeros_like(total_lag_time, dtype=np.float64)
    for i in range(len(total_lag_time)):
        lag = total_lag_time[i]
        msd = np.mean((traj[:-lag] - traj[lag:]) ** 2)
        msd_results[i] = msd
    return msd_results


class Dynamic_Langevin:
    
    def __init__(self, dt = None, frequency = 10, torque = 0, x0 = 0, gamma_cylinder = False):  
        self.KT = 300*1.3806452e-23
        self.viscosity_NS_m2 = 0.001
        self.dt = dt
        self.frequency = frequency
        self.L = 2 * np.pi / self.frequency
        self.torque = torque
        self.x0 = x0
        self.gamma_cylinder = gamma_cylinder
        self.x_pot = np.linspace(0, 2*np.pi, 50000)
        if not gamma_cylinder:
            self.R_m = 1e-6
            self.moment_inertia = (2/5)*self.m_kg*self.R_m**2
            self.m_kg = 1.1e-14 
            self.gamma = 8 * np.pi * self.viscosity_NS_m2 * self.R_m**3
        else:
            """
            Desgarceaux, R., Santybayeva, et al. (2020). High-resolution photonic force microscopy based on sharp nanofabricated tips. Nano Letters, 20(6)
            """
            self.L = 100e-9   #cylinder_length
            self.gamma = 3.841*np.pi*self.viscosity_NS_m2*self.L*self.R_m**2*(1+0.3) # [Nsm] cylinder gamma_rot_parallel for diam=0.5*length
        self.D = self.KT / self.gamma
         
       
    def make_potential_sin(self, V0=None, plots=False):
        ''' 
        periodic sinusoidal potential, ampl and tilt in KT units
        return array
        '''
        U = V0/2*self.KT*np.sin(self.x_pot*self.frequency) - self.torque*self.x_pot*self.KT/(2*np.pi)
        if plots:
            plt.figure('make_potential_sin', clear=True)
            plt.plot(self.x_pot, U/self.KT, ',')
            plt.ylabel('pot. [KT]')
        return U
    
    
    def analytical_potential(self,x,V0):
        U = V0/2*np.sin(x*self.frequency) - self.torque*x/(2*np.pi)
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
    
    
    def msd(self, traj, time_end=1/4, msd_nbpt = 500,print_time=False, log = True):
        """
        Compute the msd along a trajectory with msd_nbpt number of points log or linear distributed 
        """
        t0 = time.time()
        max_lagtime = int(len(traj) * time_end)
        if log:
            total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])
        else:
            total_lag_time = np.linspace(1,max_lagtime,msd_nbpt)
        msd_results = single_msd(traj, total_lag_time)
        if print_time==True:
            print(f'MSD_numba done in {time.time() - t0:.1f} s')
        return total_lag_time,msd_results 
    
    
    def traj_and_msd(self,repetition=None,N=None,Amplitude=None,x0=[0],ide=0,msd_nbpt = 500, time_end=1/4,save=True,remove_mean=True,log=True,id=None):
        """
        Generate trajectory and compute the MSD and repeat for an "repetition" amount of times
        return the mean of the computed MSD
        """
        t0 = time.time()
        msd_box = []
        U = self.make_potential_sin(ampl = Amplitude)
        
        for i in range(repetition):
            traj = self.main_traj_( N, Amplitude, U, x0, ide )
            if remove_mean == True:
                t = np.arange(len(traj))*self.dt
                traj = traj - t*traj[-1]/t[-1]
            total_lag_time,msd_results = self.msd(traj, time_end=time_end, msd_nbpt = msd_nbpt, print_time=False, log=log)
            msd_box.append(msd_results)
            print(f'MSD done in {time.time() - t0:.1f} s')
            del traj
            
        max_lagtime = int(N * time_end)
        mean_msd = np.concatenate(([0],np.mean(msd_box, axis=0)))
        time_axis = np.concatenate(([0], total_lag_time)) if log else np.concatenate(([0],np.linspace(1, N * time_end, msd_nbpt)))*self.dt
        
        if save == True:
            if self.gamma_cylinder:
                geometry = 1 
            else:
                geometry = 0
            if remove_mean:
                rmv = 1
            else:
                rmv = 0 
            if log:
                distrib = 1
            else:
                distrib = 0
                
            np.save(f't,msd_{N}npts,rep_{repetition},torque_{self.torque},dt={self.dt},id_{id}_geometry={geometry},removed_mean={rmv},log={distrib}',[time_axis,mean_msd])
        return time_axis,mean_msd
        

    """
    Analytical function integration
    """
    
    
    def boltzman_integrand(self,x,V0,sign):
        return np.exp(sign*self.analytical_potential(x, V0))


    def boltzman_integral(self,amplitude,sign):
        L = self.L
        result, _ = quad(self.boltzman_integrand, -L/2, L/2, args=(amplitude,sign))
        return result 


    def I_plus(self,x,V0):
        L = self.L
        result, _ = quad(self.boltzman_integrand, x - L, x, args=(V0,1))
        return (1 / self.D) * self.boltzman_integrand(x, V0, -1) * result
    
    
    def I_minus(self,x,V0):
        L = self.L
        result, _ = quad(self.boltzman_integrand, x , x + L, args=(V0,-1))
        return (1 / self.D) * self.boltzman_integrand(x, V0, 1) * result


    def D_eff_no_tilt(self, V0):
        """
        Lifson, S. & Jackson, J. L. On the self-diffusion of ions in a polyelectrolyte solution.
        The Journal of Chemical Physics 36, 2410–2414 (1962)

        Return D_eff(V0) 
        """
        L = self.L
        lifson_jackson1 = self.D *L**2 / ((self.boltzman_integral(V0,-1)) * (self.boltzman_integral(V0,1)))
        return lifson_jackson1

            
    def D_eff_tilt(self, V0):
        """
        Reimann, P. et al. (2002). Diffusion in tilted periodic potentials: Enhancement, universality, and scaling. Physical Review E. 
        Return D_eff(V0,F) 
        """
        L = self.L
        I_plus_minus_integral, _ = quad(lambda x: self.I_plus(x,V0)**2 * self.I_minus(x,V0), 0, L)
        I_minus_integral, _ = quad(lambda x: self.I_minus(x,V0), 0, L)
        D_eff = (self.D * L * I_plus_minus_integral) / (I_minus_integral**3)
        return D_eff























