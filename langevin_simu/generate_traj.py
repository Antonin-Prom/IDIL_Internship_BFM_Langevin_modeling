import numpy as np
from Class_particle_in_potential import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import numpy as np
import time


def convert_time_axis(Dt_a,dt):
    sim = DiffusionSimulation()
    t = 26*26*Dt_a/sim.rotational_einstein_diff
    N = int(t/dt)
    return t,N
    

def generate_traj(N,A,dt):
    sim = DiffusionSimulation(dt = dt)
    traj = sim.main_traj( N, A)
    #np.save(f'traj_walrus_{N:.0f}_amplitude_{A:.0f}_dt_{dt:.1f}',traj)
    return traj

def generate_traj_static(N=3600000,A=2,dt=1e-5):
    sim = DiffusionSimulation(dt)
    traj = sim.static_process( N, A, 0)
    #np.save(f'traj_static_{N:.0f}_amplitude_{A:.0f}_dt_{dt:.1f}',traj)
    return traj

"""
traj_static = np.load('traj_10000000_amplitude_2.npy',allow_pickle=True)
traj_proceed =  np.load('traj_walrus_10000000_amplitude_2_dt_0.0.npy',allow_pickle=True)
"""
sim = DiffusionSimulation(1e-5)

def run_msd_traj(traj):
    msd = sim.mean_square_displacement1(traj,0, 1/4, 100)
    return msd

npts = int(1e8)
                                                                                                                                                                       

def run_parallel(repetitions=10, n_jobs=5, Amplitude = None,dt=None):
    ''' parallel computations of long_loop() 
        parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
    '''
    # Parallel:
    print(f'run_serial_parallel(): Parallel working...')
    t0 = time.time()
    parallel_out = Parallel(n_jobs=n_jobs)(delayed(generate_traj)(N = npts,A = Amplitude , dt = dt) for i in range(repetitions))
    print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
    np.save(f'trajectories_{npts:.0f}points_amplitude_{Amplitude:.0f}kT_dt_10^-5',parallel_out)
    return parallel_out

#trajectories = np.load(f'trajectories_{npts:.0f}points_amplitude_5kT_dt_10^-4.npy',allow_pickle=True)

def calculate_msd_chunk_linear(i, nb_chunks=10, time_end=1/4, traj=None,time_skip=None):
    """

    Parameters
    ----------
    i : int
        iterator.
    nb_chunks : int
        Number of chunks in which the lag time is divided. The default is 10.
    time_end : float
        Fraction of the trajectory length on which MSD is runned. The default is 1/4.
    traj : array, 
        The angular trajectory
    time_skip : int
        Number of lag time skipped when calculating MSD

    Returns
    -------
    msd_chunk : array
        chunk of MSD for a given lag time slice

    """
    max_lagtime = int(len(traj) * time_end)
    chunks_size = int(max_lagtime / nb_chunks)
    maxchunk_list = [chunks_size * i for i in range(0, nb_chunks)]
    traj = np.unwrap(traj)
    current_max = maxchunk_list[i]
    if i == nb_chunks - 1:
        current_max = max_lagtime
    previous_max = maxchunk_list[i - 1]
    msd_chunk = [np.mean((traj[:-lag] - traj[lag:]) ** 2) for lag in range(max(1,previous_max), current_max,time_skip*i/2 + 1)]
    return msd_chunk

def calculate_msd_chunk_linear2(i, nb_chunks=10, time_end=1/4, traj= None,time_skip=None,nb_skip_first_chunk = None):
    """

    Parameters
    ----------
    i : int
        iterator.
    nb_chunks : int
        Number of chunks in which the lag time is divided. The default is 10.
    time_end : float
        Fraction of the trajectory length on which MSD is runned. The default is 1/4.
    traj : array, 
        The angular trajectory
    time_skip : int
        Number of lag time skipped when calculating MSD

    Returns
    -------
    msd_chunk : array
        chunk of MSD for a given lag time slice

    """
    max_lagtime = int(len(traj) * time_end)
    chunks_size = int(max_lagtime / nb_chunks)
    maxchunk_list = [chunks_size * i for i in range(0, nb_chunks)]
    traj = np.unwrap(traj)
    current_max = maxchunk_list[i]
    if i == nb_chunks - 1:
        current_max = max_lagtime
    previous_max = maxchunk_list[i - 1]
    if i == 1:
        time_skip = nb_skip_first_chunk
    msd_chunk = [np.mean((traj[:-lag] - traj[lag:]) ** 2) for lag in range(max(1,previous_max), current_max,time_skip)]
    return msd_chunk

def run_parallel_msd_chunk(nb_chunks=10, n_jobs=5, time_end=1/4, time_skip=1000, traj= None ,n=int,nb_skip_first_chunk = 10):
    """
    

    Parameters
    ----------
    nb_chunks : int
        Number of chunks in which the lag time is divided. The default is 10.
    n_jobs : int
        Number of workers processing in parallel. The default is 5.
    time_end : float
        Fraction of the trajectory length on which MSD is runned. The default is 1/4.
    time_skip : int
        Number of lag time skipped when calculating MSD
    traj : array, 
        The angular trajectory
    n : int
        The label of the trajectory to insert in the file name

    Returns
    -------
    final_msd : TYPE
        DESCRIPTION.

    """
    print(f'run_msd_parallel(): Parallel working...')
    t0 = time.time()
    msd_results = Parallel(n_jobs=n_jobs)(delayed(calculate_msd_chunk_linear2)(i, nb_chunks=nb_chunks, time_end=time_end, traj=traj,time_skip = time_skip,nb_skip_first_chunk=nb_skip_first_chunk) for i in range(1, nb_chunks-1))
    print(f'run_msd_parallel(): Parallel done in {time.time() - t0:.1f} s')
    final_msd = np.concatenate(msd_results)

    np.save(f'traj{n:.0f}_msdlin2_tskip1000_end4_amplitude_5kT_dt_10^-4', final_msd)
    return final_msd

def calculate_msd_chunk_log(current_chunk, nb_chunks=10, time_end=1/4, traj= None, msd_nbpt = None):
    traj = np.array(traj)
    msd_chunk = [np.mean((traj[:-lag] - traj[lag:]) ** 2) for lag in current_chunk if lag !=0]
    return msd_chunk

def run_parallel_msd_chunk_log(nb_chunks=10, n_jobs=5, time_end=1/4, msd_nbpt = None, traj=None,n=None):
    print(f'run_msd_parallel(): Parallel working...')
    t0 = time.time()
    
    traj = np.unwrap(traj)    
    max_lagtime = int(len(traj) * time_end)
    total_lag_time = [int(lag) for lag in (np.logspace(0.1,int(np.log10(max_lagtime)),msd_nbpt))]
    chunks_size = int(len(total_lag_time) / nb_chunks)
    chunk_list = []
  
    for j in range(nb_chunks-1):
        chunk_list.append(total_lag_time[int(j*chunks_size):int((j+1)*chunks_size)])
    
    msd_results = Parallel(n_jobs=n_jobs)(delayed(calculate_msd_chunk_log)(
        current_chunk, nb_chunks=nb_chunks, time_end=time_end, traj=traj, msd_nbpt=None
    ) for current_chunk in chunk_list)
    print(f'run_msd_parallel(): Parallel done in {time.time() - t0:.1f} s')
    final_msd = np.concatenate(msd_results)

    np.save(f'logmsd_traj{n:.0f}_end4_amplitude_5kT_dt_10^-4', final_msd)
    return final_msd


"""
# Load trajectories and set other parameters
trajectories = np.load('trajectories_static_100000000_amplitude_2kT_dt_10^-4.npy',allow_pickle=True)
traj = trajectories[0]

# Run the parallel MSD calculation
run_parallel_msd_chunk(nb_chunks=10, n_jobs=5, time_end=1/4, time_skip=1000, traj=traj)
"""


def linear_D(t,D,shift):
    return 2*D*t + shift

def theory_curve_oscillatory(time,A):
    simu1 = DiffusionSimulation(dt = 1e-4)
    D_eff = simu1.lifson_jackson_noforce(A)
    print(D_eff)
    msd = []
    for t in time :
        def integrand_msd(y):
            return y*y*np.exp(-simu1.tilted_periodic_potential(A, y*np.sqrt(t)))*np.exp(-y*y/(4*D_eff))/np.sqrt(4*np.pi*simu1.rotational_einstein_diff)   
        lower_limit = -5
        upper_limit = 5
        result, error = quad(integrand_msd, lower_limit, upper_limit,epsabs=1e-8, epsrel=1e-8)
        msd.append(t*result)
    return np.array(msd)

def mean_msd():
    msd_list = []
    for i in range(0,10):
        msd_list.append(np.load(f'logmsd_traj{i:.0f}_end4_amplitude_5kT_dt_10^-4.npy'))
    return np.mean(msd_list,axis=0)

"""
for i in range(0,10):
    trajectories = np.load('trajectories_100000000points_amplitude_5kT_dt_10^-4.npy',allow_pickle=True)
    run_parallel_msd_chunk_log(nb_chunks=10, n_jobs=5, time_end=1/4, msd_nbpt = 2000, traj=trajectories[i],n=i)
"""
    
    
    #(f'logmsd_traj{i:.0f)_end4_amplitude_5kT_dt_10^-4.npy)
    

"""
popt, pcov = scipy.optimize.curve_fit(linear_D, time_array, msd_test)
D_fit,shift = popt[0],popt[1]
plt.plot(time_array, 2*D_fit * time_array + shift , label = f'Linear fit, D_fit = {D_fit:.3f}, L&J coef =  {D_LJ:.3f} ')

plt.loglog(time_array,msd_test/aa, label = 'MSD for 2kT, 26 periodic potential')
plt.xlabel('Dt/a²')
plt.ylabel('<x²>/a²')
plt.legend()
plt.show()
"""

"""
plotting and comapring with theory with out std and only one MSD
t_arr = np.logspace(1,8.5,1000)*0.1#*0.1648/(26*26)
theo_msd = simu1.theory_curve_oscillatory(t_arr,5)
plt.loglog(t_arr,theo_msd, label = 'theory_MSD for 5kT, 26 periodic potential')
plt.loglog(time_array,msd_test, label = 'MSD for 5kT, 26 periodic potential')
plt.loglog(t_arr, linear_D(t_arr,D_LJ,0), label = f'L&J 2D*t, D* = {D_LJ:.5f}')
plt.xlim(1,10**3)

msd0 = np.load('traj1_msd_traj0_100000000_tskip1000_end4_amplitude_5kT_dt_10^-4.npy')
msd42 = np.load('traj42_msd_traj0_100000000_tskip1000_end4_amplitude_5kT_dt_10^-4.npy')
t42 = np.logspace(0,log10(25000000*1e-4),len(msd42))
t0 = np.arange(0,len(msd0))*0.1
theo_msd_42_err5 = theory_curve_oscillatory(t42,5)

fig, ax = plt.subplots(figsize = (9, 6))
ax.scatter(t42, msd42, s=0.5, label = 'Simulated MSD for 5kT, 26 periodic potential')
ax.scatter(t42, theo_msd, s=0.5, label = 'Simulated MSD for 5kT, 26 periodic potential')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('t[s]')
ax.set_ylabel('<x²> [rad²]')
plt.legend()




theo_msd_5 = theory_curve_oscillatory(t0,5)
fig, ax = plt.subplots(figsize = (9, 6))
ax.scatter(t0, uni_mean, s=0.5, label = 'Simulated MSD_mean10 for 5kT')
ax.scatter(t0, theo_msd_5, s=0.5, label = 'Theoretical MSD for 5kT,')
ax.scatter(t0,linear_D(t0,0.0002221,0), s=0.5 label = '2d*t')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('t[s]')
ax.set_ylabel('<x²> [rad²]')
plt.legend()




"""
