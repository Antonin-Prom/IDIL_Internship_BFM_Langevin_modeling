# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:56:43 2024

@author: CBS
"""
from Class_particle_v2_periodic_potential import *
from scipy.stats import norm
from scipy.optimize import curve_fit


        
def qth_moment(q,t,D_eff):
    t *= (0.1648)/(2*np.pi/26)**2
    return((2**q)/(np.sqrt(np.pi)))*(D_eff*t)**(q/2)*scipy.special.gamma((1+q)/2)
    
def second_moment(t,D_eff):
    return(4/(np.sqrt(np.pi)))*(D_eff*t)*scipy.special.gamma(3/2)

def generate_theory_msd(absciss,D_fit):
    theoretical_msd = []
    for t in absciss:
        msd_theo = qth_moment(2,t,D_fit)
        theoretical_msd.append(msd_theo)
    return theoretical_msd

def generate_trajs_and_msd():
    A_list = [0,1,2,3,4,5]
    analy = DiffusionSimulation2(dt=1e-4,x0 = 0)
    for A in A_list :
        trajs = analy.run_parallel(n_jobs=-1, repetitions = 30, npts = int(1e8), Amplitude = A, torque = 0) 
        time_axis,mean_msd = analy.mean_msd_and_time_axis(trajs, n_jobs=-1, time_end=1/4, msd_nbpt = 2000, nb_traj = 30)
        np.save(f't,msd,dt={analy.dt_s},{A}kT',[time_axis*analy.dt_s,mean_msd])
    
def asdefaveri_msd(A,dt):
    p = DiffusionSimulation2(dt=dt)
    a = 2*np.pi/p.frequency
    D = p.rotational_einstein_diff           
    D_eff = p.lifson_jackson_noforce(A)
    
    def defaveri_stationary(A):
        X = np.linspace(0,a,1000)
        Y = boltz_exp(A,X)
        Z = np.mean(Y)
        return (np.mean((X**2)*Y)/Z)   
     
    def boltz_exp(A,x):
        return np.exp(-p.tilted_periodic_potential(A, x))
        
    def linear_D(t,D,shift):
        return (2*D*t + shift)/(a*a)
    
    t, msd = np.load(f't,msd,dt={dt},{A}kT.npy')
    absciss = 2*D*t
    norm_msd = msd/(a*a)
    
    min_absciss_value = 1  # Adjust this value as needed
    
    window_indices = np.where(absciss > min_absciss_value)[0]
    absciss_window = absciss[window_indices]
    norm_msd_window = norm_msd[window_indices]
    
    popt, pcov = scipy.optimize.curve_fit(linear_D, absciss_window, norm_msd_window,)
    D_fit, shift = popt[0], popt[1]
    print('D_fit = ',D_fit*2*D)
    print('D_eff = ',D_eff)
    stationary_state = defaveri_stationary(A) 
    return absciss,norm_msd,D_fit,D_eff,shift,stationary_state,a

def plot_msd_asdefvari(A,dt):
    absciss,norm_msd,D_fit,D_eff,shift,stationary_state,a = asdefaveri_msd(A,dt)
    
    def linear_D(t,D,shift):
        return (2*D*t + shift)/(a*a)
    
    plt.loglog(absciss,norm_msd,label = f'A = {A} kT')
    if A == 6:
        plt.loglog(absciss,stationary_state*np.ones(len(absciss)),linestyle ='--')
    #plt.loglog(absciss[int(len(absciss)/2):],linear_D(absciss,D_eff,shift)[int(len(absciss)/2):],linestyle ='--',color='green')
    frac = 1.5
    #plt.loglog(absciss[int(len(absciss)/frac):],generate_theory_msd(absciss,D_eff)[int(len(absciss)/frac):],color='red')
    #D_eff/(2*0.1648)
    plt.loglog(absciss[int(len(absciss)/frac):],linear_D(absciss,D_fit,shift)[int(len(absciss)/frac):],linestyle ='--',color='black') #[int(len(absciss)/4):]
"""
for A in range(0,7,2):
    plot_msd_asdefvari(A,0.001)
plt.xlabel('Dt/L²')
plt.ylabel(r'$\theta$/L²')
plt.legend()
plt.show()
"""