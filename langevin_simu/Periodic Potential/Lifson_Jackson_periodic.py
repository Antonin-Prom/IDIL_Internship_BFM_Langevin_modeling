# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:52:35 2024

@author: CBS
"""
from Class_particle_v2_periodic_potential import *
from scipy.stats import norm
from scipy.optimize import curve_fit

p = DiffusionSimulation2(dt = 0.001,frequency=10)
def asdefaveri_msd(A,dt):
    a = 2*np.pi/p.frequency
    D = p.rotational_einstein_diff           
    D_eff = p.lifson_jackson_noforce(A)
    
    def linear_D(t,D,shift):
        return (2*D*t + shift)
    
    t, msd = np.load(f'LJ_t,msd,dt={p.dt_s},{A}kT.npy')
    absciss = t 
    popt, pcov = scipy.optimize.curve_fit(linear_D, absciss, msd)
    D_fit, shift = popt[0], popt[1]
    frac = 2
    #plt.plot(absciss,msd,label = f'A = {A} kT')
    #plt.plot(absciss,linear_D(absciss,D_fit,shift),linestyle ='--',color='black') 
    #plt.plot(absciss,linear_D(absciss,D_eff,shift),linestyle ='--',color='red') 

    print('D_fit = ',D_fit)
    print('D_eff = ',D_eff)
    
    return D_fit,D_eff,pcov

A_list1_5 = np.logspace(0,np.log10(5),50)
A_list0_1 = np.linspace(0,1,20)


results = []
results0_1 = []
def load_and_D():
    for A in A_list0_1:
        trajs = p.run_parallel(repetitions=50, n_jobs=5, npts = 1000000, Amplitude = A, torque = 0)
        time_axis,mean_msd,_,_ = p.mean_msd_and_time_axis(trajs, n_jobs=5, time_end=1/4, msd_nbpt = 2000, nb_traj = 50)
        time_axis *= p.dt_s
        np.save(f'LJ_t,msd,dt={p.dt_s},{A}kT',[time_axis,mean_msd])
        D_fit,D_eff,_ = asdefaveri_msd(A,0.001)
        results0_1.append([D_fit,D_eff])#pcov[0]
    np.save('LJ_D_fit,D_eff,0_1',results0_1)#pcov
load_and_D()
"""
D_fit = []
D_eff = []
[D_fit.append(results[:][i][0]) for i in range(len(A_list))]
[D_eff.append(results[:][i][1]) for i in range(len(A_list))]
print(results)
print(D_fit)

plt.plot(A_list,D_fit)
plt.plot(A_list,D_eff)
plt.show()
"""