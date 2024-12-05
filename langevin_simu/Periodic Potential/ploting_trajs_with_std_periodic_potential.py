# -*- coding: utf-8 -*-
"""b
Plots the traj with variance
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import numpy as np
import time
"""
trajs5 = np.load(f'trajectories_100000points_amplitude_0kT_dt_1e-05_torque_0.npy')
std5 = np.std(trajs5, axis=0)
"""

D = 0.16480238117707385


    
def plot_trajs(j):
    data = [np.load(f'trajectories_10000000,nb_traj_20points_amplitude_{A}kT,frequency_10_dt_0.0001_torque_0kT.npy') for A in range(1,10)]
    print('aaaaaaaaaaaaaaaaaa',len(data))
    trajs = data[0][j]
    X = 1000
    t = np.arange(0,len(trajs[0]))*1e-4
    for A in range(5,9):
        plt.plot(t[:X],trajs[A][:X],linewidth =0.5,label=f'{A}kT')
        plt.plot(t[:X],np.zeros(X),color='red')
    plt.xlabel('time [s]')
    plt.ylabel('displacement [rad]')
    plt.legend()
    plt.show()


def mean_velocity(trajs):
    mean_vel = []
    for traj in trajs:
        v = []
        [v.append(abs(traj[i+2]-traj[i])) for i in range(len(traj)-2)]
        mean_vel.append(v)
        
    mean_vel = np.mean(mean_vel,axis=0)
    return mean_vel

def why_stonks(X):
    trajs8l = np.load(f'trajectories_10000,nb_traj_35points_amplitude_7kT,frequency_10_dt_0.0001_torque_0kT.npy')
    m_sto = []
    m_det = []
    m_traj = []
    for traj in trajs8l:
        mean_traj = np.mean(traj)
        mean_cum_stocha = np.mean(np.cumsum(traj[1][:X]))
        mean_cum_det = np.mean(np.cumsum(traj[2][:X]))
        m_sto.append(mean_cum_stocha)
        m_det.append(mean_cum_det)
        m_traj.append(mean_traj)
    plt.plot(m_sto,color='green')
    plt.plot(m_det)
    plt.plot(m_traj,color='red')
    print(np.mean(m_sto))
    print(np.mean(m_det))
    
def plot_velocity():
    X = 1000
    trajectories = [np.load(f'trajectories_1000000,nb_traj_10points_amplitude_{A}kT,frequency_10_dt_0.0001_torque_0kT.npy') for A in range(5)]
    for trajs in trajectories:
        mvel = mean_velocity(trajs[:X])
        t = np.arange(0,len(mvel))*1e-4
        print(np.mean(mvel))
        #plt.plot(t,mvel,linewidth =0.5,label='A = 0kT')
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        