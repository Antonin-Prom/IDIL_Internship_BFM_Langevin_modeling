import numpy as np
from Class_particle_in_potential import *
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit


einstein_diff = 0.164848


data = np.load('msd_no_torque.npy',allow_pickle=True)
loaded_dict = data[()]
time = loaded_dict['time']
std = loaded_dict['std']
msd = loaded_dict['msd']
D_eff = loaded_dict['D_eff']



def theory_plot():
    sim = DiffusionSimulation()
    t_arr = np.logspace(-3,3,2000)
    theo_msd = sim.theory_curve_oscillatory(t_arr,10)/(26*26)
    popt, pcov = scipy.optimize.curve_fit(linear_D, t_arr, theo_msd)
    D_fit,shift = popt[0],popt[1]
    fit = linear_D(t_arr, D_fit,shift)
    
     #plt.plot(t_arr, linear_D(t_arr, D_eff[0],shift=0), label= f' y = 2Dt, Fitted diffusion coefficient = {D_fit:.3f}, D_LJ = {D_eff[0]:.3f}')
    plt.plot(t_arr,theo_msd, label = 'theory defaveri')
    plt.legend()
    plt.xlabel('Dt/a²')
    plt.ylabel('MSD')
    
def linear_D(t,D,shift):
    return 2*D*t + shift

def oscillating_term(x,torque,A):
    frequency = 26
    return np.exp(torque*x + A*np.cos(x*frequency))

def plot_justify_longtime():
    x1 = np.arange(0,100000,100)
    x2 = np.arange(0,1000000,100)
    fig, axs = plt.subplots(2, 2)
    
    axs[0,0].plot(x1,oscillating_term(x1,torque=0,A=6))
    axs[0,0].set_xlabel('time [u.a]')
    
    axs[0, 1].plot(x2,oscillating_term(x2,torque=0,A=6))
    axs[0, 1].set_xlabel('time [u.a]')
    axs[0, 0].set_ylabel('exp(V(x)) no torque [u.a]')
    
    x3 = np.linspace(0,10,10000)
    x4 = np.linspace(0,30,10000)
    
    axs[1, 0].plot(x3,oscillating_term(x3,torque=1,A=6))
    axs[1, 0].set_xlabel('time [u.a]')
    axs[1, 0].set_ylabel('exp(V(x))  torque = 1 [u.a]')
    axs[1, 1].plot(x4,oscillating_term(x4,torque=1,A=6))
    axs[1, 1].set_xlabel('time [u.a]')
    
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
    
def integrant_term(y,t,A):
    part = DiffusionSimulation()
    D_LJ = part.lifson_jackson_noforce(A)
    return y*y*np.exp(-part.tilted_periodic_potential(A, y*np.sqrt(t)))*np.exp(-y*y/(4*D_LJ))/np.sqrt(4*np.pi*part.rotational_einstein_diff)


def theory_curve_oscillatory(time,A,freq):
    part = DiffusionSimulation()
    D_LJ = part.lifson_jackson_noforce(A)
    msd = []
    for t in time :
        def integrand_msd(y):
             return y*y*np.exp(-A*np.cos(y*np.sqrt(t)*freq))*np.exp(-y*y/(4*D_LJ))/np.sqrt(4*np.pi*part.rotational_einstein_diff) 
        lower_limit = -3
        upper_limit = 3
        result, error = quad(integrand_msd, lower_limit, upper_limit,epsabs=1e-8, epsrel=1e-6)
        msd.append(t*result)
    return np.array(msd)

def theory_curve_oscillatory_unit(t,A):
    part = DiffusionSimulation()
    D_LJ = part.lifson_jackson_noforce(A)
    msd = []
    
    def integrand_msd(y):
         return y*y*np.exp(-part.tilted_periodic_potential(A, y*np.sqrt(t)))*np.exp(-y*y/(4*D_LJ))/np.sqrt(4*np.pi*part.rotational_einstein_diff) 
    lower_limit = -5
    upper_limit = 5
    integ, error = quad(integrand_msd, lower_limit, upper_limit,epsabs=1e-9, epsrel=1e-9)     
    msd = t*integ
    return msd


def plot_integral_theory():
    aa = (2*np.pi/26)**2
    K = 0.1648/(aa)
    time_dta = np.logspace(-4,7,400)*K
    A = 1
    msd = theory_curve_oscillatory(time_dta,A)
    plt.xlim(0.001,1000)
    plt.ylim(0.001,50)
    plt.loglog(time_dta,msd/aa,label = 'V_0/kT = 2')
    plt.xlabel('Dt/a²')
    plt.ylabel('<x²>/a²')
    
    params, covariance = curve_fit(linear_D, time_dta, msd)
    D = params[0]
    plt.plot(time_dta,linear_D(time_dta, params[0], params[1]),label = f'fit, {D:.5f}')
    
def plot_integrand(t,A):
    y = np.linspace(-2,2,1000000)
    plt.plot(y,integrant_term(y,t,A))
    plt.xlabel('y')
    plt.ylabel('u.a')
    plt.title(f'integrand at time {t:}s, amplitude = 6kT')


def quadratic(a,b,x):
    return a*x*x + b*x


def msd_integral_with_torque(time,A,torque):
    part = DiffusionSimulation(torque=torque)
    D_LJ = part.lifson_jackson_noforce(A)
    msd = []
    for t in time :
        def integrand_with_torque(y,A,torque):
            return y*y*np.exp(-torque*y*np.sqrt(t))*np.exp(-part.tilted_periodic_potential(A, y*np.sqrt(t)))*np.exp(-y*y/(4*D_LJ))/np.sqrt(4*np.pi*part.rotational_einstein_diff)
        lower_limit = -2
        upper_limit = 2
        result, error = quad(integrand_with_torque, lower_limit, upper_limit,epsabs=1e-8, epsrel=1e-6)
        msd.append(t*result)
    return np.array(msd)

def plot_msd_integral_with_torque():
    torque = 5
    aa = (2*np.pi/26)**2
    K = 0.1648/(aa)
    time_dta = np.logspace(-4,7,100)*K
    A = 1
    msd = msd_integral_with_torque(time_dta,A,torque)
    plt.loglog(time_dta,msd/aa,label = 'V_0/kT = 2')
    plt.xlabel('Dt/a²')
    plt.ylabel('<x²>/a²')
    
    params, covariance = curve_fit(linear_D, time_dta, msd)
    D = params[0]
    plt.plot(time_dta,linear_D(time_dta, params[0], params[1]),label = f'fit, {D:.5f}')
    
def integrand_with_torque(y,A,torque,t):
    part = DiffusionSimulation(torque=torque)
    D_LJ = part.lifson_jackson_noforce(A)
    return y*y*np.exp(-torque*y*np.sqrt(t))*np.exp(-part.tilted_periodic_potential(A, y*np.sqrt(t)))*np.exp(-y*y/(4*D_LJ))/np.sqrt(4*np.pi*part.rotational_einstein_diff)

def plot_integrand_torque(t,A,torque):
    y = np.linspace(-5,5,10000)
    plt.plot(y,integrand_with_torque(y,A,torque,t))
    plt.xlabel('y')
    plt.ylabel('u.a')
    plt.title(f'Integrand at time {t:}s, amplitude = {A}kT,torque = {torque}kT')

freq = 10
tim = np.logspace(-3,4,200)
defav5kt = theory_curve_oscillatory(tim,5,freq)
plt.loglog(tim,defav5kt,label = 'Defaveri V_0/kT = 5')
plt.legend()
plt.ylim(0.001,50)
print(msd)
plt.show()
