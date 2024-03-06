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
                                                                                                                                                                       
def run_serial(repetitions=10):
    ''' serial computations of long_loop() '''
    # Serial:
    t0 = time.time()
    serial_out = np.zeros((repetitions, npts))
    for i in range(repetitions):
        print(f'run_serial_parallel(): Serial working... {i+1}/{repetitions}', end='\r' if i != repetitions-1 else '\n')
        serial_out[i, :] = long_loop(npts=npts, dt=0.001)
    # postprocessing using serial_out
    print(f'run_serial_parallel(): Serial done in {time.time() - t0:.1f} s')
    return serial_out


def run_parallel(repetitions=10, n_jobs=5):
    ''' parallel computations of long_loop() 
        parallel_out is a list (of len repetitions) of arrays "x(t)" (each of len npts)
    '''
    # Parallel:
    print(f'run_serial_parallel(): Parallel working...')
    t0 = time.time()
    parallel_out = Parallel(n_jobs=n_jobs)(delayed(generate_traj)(N = npts,A = 5 , dt = 1e-4) for i in range(repetitions))
    print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
    np.save(f'trajectories_{npts:.0f}points_amplitude_5kT_dt_10^-4',parallel_out)
    return parallel_out

#trajectories = np.load(f'trajectories_{npts:.0f}points_amplitude_5kT_dt_10^-4.npy',allow_pickle=True)

def calculate_msd_chunk(i, nb_chunks=10, time_end=1/4, traj=None,time_skip=None):
    max_lagtime = int(len(traj) * time_end)
    chunks_size = int(max_lagtime / nb_chunks)
    maxchunk_list = [chunks_size * i for i in range(0, nb_chunks)]
    print(maxchunk_list)
    traj = np.unwrap(traj)
    current_max = maxchunk_list[i]
    if i == nb_chunks - 1:
        current_max = max_lagtime
    previous_max = maxchunk_list[i - 1]
    msd_chunk = [np.mean((traj[:-lag] - traj[lag:]) ** 2) for lag in range(max(1,previous_max), current_max,time_skip)]
    return msd_chunk

def run_parallel_msd_chunk(nb_chunks=10, n_jobs=5, time_end=1/4, time_skip=1000, traj=None):
    print(f'run_msd_parallel(): Parallel working...')
    t0 = time.time()
    msd_results = Parallel(n_jobs=n_jobs)(delayed(calculate_msd_chunk)(i, nb_chunks=nb_chunks, time_end=time_end, traj=traj,time_skip = time_skip) for i in range(1, nb_chunks-1))
    print(f'run_msd_parallel(): Parallel done in {time.time() - t0:.1f} s')
    final_msd = np.concatenate(msd_results)

    np.save(f'msd_traj0_100000000_tskip1000_end4_amplitude_5kT_dt_10^-4', final_msd)
    return final_msd

"""
# Load trajectories and set other parameters
trajectories = np.load('trajectories_static_100000000_amplitude_2kT_dt_10^-4.npy',allow_pickle=True)
traj = trajectories[0]

# Run the parallel MSD calculation
run_parallel_msd_chunk(nb_chunks=10, n_jobs=5, time_end=1/4, time_skip=1000, traj=traj)
"""

"""
def linear_D(t,D,shift):
    return 2*D*t + shift

aa = (2*np.pi/26)**2
simu1 = DiffusionSimulation(frequency = 26,torque  = 0 )
D_LJ = simu1.lifson_jackson_noforce(2)
msd_test = np.load('msd_traj0_100000000_tskip1000_end4_amplitude_2kT_dt_10^-4.npy',allow_pickle=True)
time_array = np.arange(len(msd_test))*0.1*0.1648/aa

popt, pcov = scipy.optimize.curve_fit(linear_D, time_array, msd_test)
D_fit,shift = popt[0],popt[1]
plt.plot(time_array, 2*D_fit * time_array + shift , label = f'Linear fit, D_fit = {D_fit:.3f}, L&J coef =  {D_LJ:.3f} ')

plt.loglog(time_array,msd_test/aa, label = 'MSD for 2kT, 26 periodic potential')
plt.xlabel('Dt/a²')
plt.ylabel('<x²>/a²')
plt.legend()
plt.show()

t_arr = np.logspace(0.001,3,1000)*0.1*0.1648/(26*26)
theo_msd = simu1.theory_curve_oscillatory(t_arr,2)
plt.loglog(t_arr,theo_msd, label = 'theory_MSD for 2kT, 26 periodic potential')
plt.loglog(time_array,msd_test, label = 'MSD for 2kT, 26 periodic potential')
plt.legend()
"""
