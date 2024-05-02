# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:52:27 2024
r
@author: CBS
"""
from Class_particle_v2_periodic_potential import *
from scipy.stats import norm
from scipy.optimize import curve_fit


def compare_dt_kT(Ampli):
    total_time = int(1e8*1e-5)
    for dt in [1e-4,5e-4,1e-3,5e-3,1e-2,5e-2]:
        p = DiffusionSimulation2(dt=dt)
        print(p.dt_s)
        nb_points = int(total_time/dt)
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


def compare_dt_msd_given_amplitude(Ampli):
    dt_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    msd_curves = []
    time_axes = []
    for dt in dt_list:
        # Load data for each dt
        t, msd = np.load(f't,msd,dt={dt},{Ampli}kT.npy')

        msd_curves.append(msd)
        time_axes.append(t)
        
        
    mother_array = []
    for i in range(len(msd_curves)):
        for j in range(len(time_axes[i])):
           mother_array.append([time_axes[i][j],msd_curves[i][j]])
    ascending_mother = sorted(mother_array, key=lambda x: x[1])
    
    global_msd = []
    global_time = []
    for j in ascending_mother:
        global_msd.append(j[1])
        global_time.append(j[0])
    plt.scatter(global_time,global_msd, s = 0.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('time [s]')
    plt.ylabel('mean square displacement')