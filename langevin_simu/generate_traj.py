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
    sim = DiffusionSimulation(dt)
    traj = sim.proceed_traj1( N, A)
    #np.save(f'traj_walrus_{N:.0f}_amplitude_{A:.0f}_dt_{dt:.1f}',traj)
    return traj

def generate_traj_static(N=3600000,A=2,dt=1e-5):
    sim = DiffusionSimulation(dt)
    traj = sim.static_process( N, A)
    #np.save(f'traj_static_{N:.0f}_amplitude_{A:.0f}_dt_{dt:.1f}',traj)
    return traj
    
traj_static = np.load('traj_10000000_amplitude_2.npy',allow_pickle=True)
traj_proceed =  np.load('traj_walrus_10000000_amplitude_2_dt_0.0.npy',allow_pickle=True)

sim = DiffusionSimulation(1e-5)

def run_msd_traj(traj):
    msd = sim.mean_square_displacement1(traj,0, 1/4, 100)
    return msd

npts = 3600000

                                                                                                                                                                       
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
    parallel_out = Parallel(n_jobs=n_jobs)(delayed(generate_traj)(N = npts,A = 2 , dt = 1e-4) for i in range(repetitions))
    print(f'run_serial_parallel(): Parallel done in {time.time() - t0:.1f} s')
    return parallel_out






