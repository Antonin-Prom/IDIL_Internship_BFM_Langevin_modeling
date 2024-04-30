# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:52:35 2024

@author: CBS
"""
from Class_particle_v2_periodic_potential import *
from scipy.stats import norm
from scipy.optimize import curve_fit


def asdefaveri_msd(A,dt):
    p = DiffusionSimulation2(dt=dt)
    a = 2*np.pi/p.frequency
    D = p.rotational_einstein_diff           
    D_eff = p.lifson_jackson_noforce(A)
    
    def linear_D(t,D,shift):
        return (2*D*t + shift)/(a*a)
    
    t, msd = np.load(f't,msd,dt={dt},{A}kT.npy')
    absciss = t
    popt, pcov = scipy.optimize.curve_fit(linear_D, absciss)
    D_fit, shift = popt[0], popt[1]
    print('D_fit = ',D_fit*2*D)
    print('D_eff = ',D_eff)
    
    return D_fit,D_eff,pcov

inverse = np.linspace(0.0001,10,10)
A_list = 1/inverse

p = DiffusionSimulation2(dt = 0.001,frequency=10)

results = []
for A in A_list:
    trajs = p.run_parallel(repetitions=10, n_jobs=5, npts = 1000000, Amplitude = A, torque = 0)
    time_axis,mean_msd,std,matrix = p.mean_msd_and_time_axis(trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = 10)
    np.save(f'LJ_t,msd,dt={p.dt_s},{A}kT',[time_axis*p.dt_s,mean_msd])
    D_fit,D_eff,pcov = asdefaveri_msd(A,0.001)
    results.append[D_fit,D_eff,pcov]
    np.save('LJ_D_fit,D_eff,pcov',results)
    
plt.plot(A_list,D_fit,'Numerical D')
plt.plot(A_list,D_eff,'Theoretical D')
plt.show()