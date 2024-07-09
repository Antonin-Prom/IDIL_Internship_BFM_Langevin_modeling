
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
from matplotlib.cm import viridis
label_fontsize = 18
legend_fontsize=14
viridis = plt.cm.viridis

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
    msd_results = np.ones_like(total_lag_time, dtype=np.float64)
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
        self.gamma = 8 * np.pi * self.viscosity_NS_m2 * self.R_m**3
        self.L = 100e-9   #cylinder_length
        #self.gamma = 3.841*np.pi*self.viscosity_NS_m2*self.L*self.R_m**2*(1+0.3) # [Nsm] cylinder gamma_rot_parallel for diam=0.5*length
        self.moment_inertia = (2/5)*self.m_kg*self.R_m**2
        self.tau = self.moment_inertia / self.gamma
        self.D = self.KT / self.gamma
        self.dt = dt # tau
        self.frequency = frequency
        self.space_step = 1e-8
        self.torque = torque#/(2*np.pi)
        self.x0 = x0
        self.analytical = analytical
        self.x_pot = np.linspace(0, 2*np.pi, 50000)
        self.effective_velocity = (self.torque/(2*np.pi)*self.KT)/self.gamma
    

    
    def configure_plots(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [8, 6]
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['grid.color'] = 'gray'
        plt.rcParams['grid.alpha'] = 0.5
        
    def make_potential_sin(self, ampl=None, plots=False):
        ''' 
        periodic sinusoidal potential, ampl and tilt in KT units
        return array
        '''
        U = ampl/2*self.KT*np.sin(self.x_pot*self.frequency) - self.torque*self.x_pot*self.KT/(2*np.pi)
        if plots:
            plt.figure('make_potential_sin', clear=True)
            plt.plot(self.x_pot, U/self.KT, ',')
            plt.ylabel('pot. [KT]')
        return U
    
    def analytical_potential(self,x,A):
        U = A/2*np.sin(x*self.frequency) - self.torque*x/(2*np.pi)
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
    
    def msd(self, traj, time_end=1/4, msd_nbpt=40, print_time=False, distrib_log=False):
        traj = np.unwrap(traj)
        
        t0 = time.time()
        max_lagtime = int(len(traj) * time_end)
        if distrib_log:
            total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, np.log10(max_lagtime), msd_nbpt))])
        else:
            total_lag_time = np.linspace(1, max_lagtime, msd_nbpt, dtype=int)
        
        msd_results = single_msd(traj, total_lag_time)

        if print_time:
            print(f'MSD_numba done in {time.time() - t0:.1f} s')
        
        return msd_results, total_lag_time  
    
    
    def brutal_msd(self, repetition=None, N=None, Amplitude=None, x0=[0], ide=0, msd_nbpt=40, time_end=1/4, save=True, remove_mean=False, distrib_log=False):
        t0 = time.time()
        msd_box = []
        U = self.make_potential_sin(ampl=Amplitude) 
        for i in range(repetition):
            traj = self.main_traj_(N, Amplitude, U, x0, ide)
            if remove_mean:
                t = np.arange(len(traj)) * self.dt
                traj = traj - t * traj[-1] / t[-1]
            
            msd_results, total_lag_time = self.msd(traj, time_end=time_end, msd_nbpt=msd_nbpt, print_time=False, distrib_log=distrib_log)
            msd_box.append(msd_results)
            
            print(f'MSD done in {time.time() - t0:.1f} s')
            del traj
        
        mean_msd = np.concatenate(([0], np.mean(msd_box, axis=0)))
        time_axis = np.concatenate(([0], total_lag_time)) if distrib_log else np.concatenate(([0],np.linspace(1, N * time_end, msd_nbpt)))*self.dt

        if save:
            np.save(f't,msd_{N}npts_{repetition}rep_torque_{self.torque}kT_dt={self.dt},bead,removed_mean', [time_axis, mean_msd])
        
        return time_axis, mean_msd
    
    
    def brutal_msd_amplitude_range(self,ampl_range=[0,4,6,18],repetition=None,N=None,x0=[0],ide=0,msd_nbpt = 500, time_end=1/4,save=True):
        time_box,mean_msd_box = [],[]
        for A in ampl_range:
            time_axis,mean_msd = self.brutal_msd(repetition=repetition,N=N,Amplitude=A,x0=x0,ide=ide,msd_nbpt = msd_nbpt, time_end=time_end,save=save)
            time_box.append(time_axis)
            mean_msd_box.append(mean_msd)
        np.save(f'A=0,4,6,18_t,msd_{N}npts_{repetition}rep_torque_{self.torque}kT_dt={self.dt},bead',[time_box,mean_msd_box])
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
    
    
    """
    Lifson and Jackson methods
    """

    def integrand1(self, x, amplitude):
        return np.exp((amplitude/2)*np.sin(x*self.frequency)-self.torque/(2*np.pi))
    
    def full_integrand(self,x,amplitude):
        return np.exp(-self.analytical_potential(x, amplitude))
    
    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / self.frequency, np.pi / self.frequency, args=(amplitude))
        return result
    
    def full_factor(self,amplitude):
        a = 2 * np.pi / self.frequency
        result, _ = quad(self.full_integrand, -0, a, args=(amplitude))
        return result    
    
    def lifson_jackson(self, amplitude): #Meet einstein coeff at 0 barrier
        if self.torque == 0:
            lifson_jackson1 = self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))
            return lifson_jackson1
        else:
            F = self.torque/(2*np.pi) #KT already taken in acount            
            lifson_jackson2 = (self.D * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))) * ((np.sinh(F*(2*np.pi / self.frequency)/2))/(F*(2*np.pi / self.frequency)/2))**2
            return lifson_jackson2            
 
    def D_eff_reimann(self,amplitude):
        a = 2 * np.pi / self.frequency
        A = amplitude
        
        def I_plus(x):
            result, _ = quad(self.full_integrand, x - a, x, args=(A))
            return (1 / self.D) * self.full_integrand(x, -A) * result
        
        def I_minus(x):
            result, _ = quad(self.full_integrand, x , x + a, args=(-A))
            return (1 / self.D) * self.full_integrand(x, A) * result
        
        
        I_plus_integral, _ = quad(lambda x: I_plus(x)**2 * I_minus(x), 0, a)
        I_minus_integral, _ = quad(I_minus, 0, a)
        
        D_f = (self.D * (2 * np.pi / a) * I_plus_integral) / (I_minus_integral**3)
        
        return D_f
        
    

A = 14
dt=5e-5
N = int(5e7)
# --> Plateau entre 300 et 340_000 pt (a multiplier par dt)
repetitions = 10
msd_nbpt = 100
x0 = np.zeros(repetitions)
obj = LangevinSimulator(dt=dt,torque=0,frequency=10)
ide = 0
trajs = []
for i in range(repetitions):
    U = obj.make_potential_sin(ampl=A) 
    traj = np.unwrap(obj.main_traj_(N, A, U, x0, ide))
    trajs.append(traj)
np.save(f'trajs_A={A}_dt={dt}_N={N}_rep={repetitions}_defaveri_plateau',trajs)

trajs = np.load(f'trajs_A={A}_dt={dt}_N={N}_rep={repetitions}_defaveri_plateau.npy')
lag_N_begin = 300
lag_N_end = 340000
lag_time = np.linspace((lag_N_begin),(lag_N_end),msd_nbpt)

def mean_msd_in_lag_range():
    msd_box = []
    for traj in trajs:
        msd_for_one_traj = single_msd(traj, lag_time)
        msd_box.append(msd_for_one_traj) 
    mean_msd = np.mean(msd_box, axis=0)
    return mean_msd
mean_msd = mean_msd_in_lag_range()


@njit
def matrix_at_t(trajs, t):
    m = []
    for traj in trajs:
        m.append(traj[t])
    return np.array(m)

time_select = lag_time
matrixes = [matrix_at_t(trajs, t) for t in time_select]

all_data = np.concatenate(matrixes)
plt.figure(figsize=(10, 6))
colors = viridis(np.linspace(0, 1, len(time_select)))
labels = [f't = {t * dt:.0f}s, msd = {mean_msd[idx]}' for idx,t in enumerate(time_select)]

num_bins = 100
for idx, (m, color) in enumerate(zip(matrixes, colors)):
    hist_data, bins = np.histogram(m, bins=num_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, hist_data, 'o', color=color, label=labels[idx])



#t,msds = obj.brutal_msd(repetition=repetitions,N=N,Amplitude=A,x0=[0],ide=0,msd_nbpt = msd_nbpt, time_end=1/3,save=True,remove_mean=False,distrib_log=True)

plt.xscale('log')
plt.yscale('log')
plt.show()


def fit_bellour():
    p = LangevinSimulator(dt=1e-4, frequency=10)
    a = 2 * np.pi / p.frequency

    # Corrected bellour_fit function to return MSD


    # Load data
    t_box, msd_box = np.load('A=0,4,6,18_t,msd_10000000npts_10rep_torque_0kT_dt=0.0001,bead.npy')
    A_box = [0, 4, 6, 18]
    colors = viridis(np.linspace(0, 1, len(A_box)))

    plt.figure()
    for idx, (t, msd) in enumerate(zip(t_box, msd_box)):
        print(idx)
        t *= 1e-4
        color = colors[idx]

        D_LJ = p.lifson_jackson(A_box[idx])
        print('D_LJ', D_LJ)
        
        p.fit_msd_Bellour( msd_time = t, msd = msd, plots=True, plots_clear=True)

    plt.xlabel('Lag time (s)', fontsize=12)
    plt.ylabel('MSD (rad²)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('periodic_msd.png', dpi=300)
    plt.show()













