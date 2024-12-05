from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Langevin_Class_v2 import *
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import jv
from scipy.integrate import simps

p = LangevinSimulator(dt=0.0001, frequency=10, torque=0)
amplitude = 6
D_eff = p.lifson_jackson(amplitude)
D0 = p.D


def integral_at_t(t):
    
    def integrand(y, amplitude):
        return y**2 * p.integrand1(-y * np.sqrt(t), amplitude) * np.exp(-(y**2) / (4 * D_eff)) / np.sqrt(4 * np.pi * D0)
    
    lim = 4
    low_lim = -lim*np.pi / np.sqrt(t)
    high_lim = lim*np.pi / np.sqrt(t)
    
    angle_space = np.linspace(-lim*np.pi, lim*np.pi, 10000)
    y = angle_space / np.sqrt(t)
    
    def calculate_msd(t, amplitude):
        integrand_func = lambda y: integrand(y, t, amplitude)
        integral_value, _ = quad(integrand_func, -np.pi, np.pi)
        return 2 * D_eff * t * integral_value

    def results(amplitude):  
        result, _ = quad(integrand, low_lim, high_lim, args=(amplitude,))
        return result    
    return t * results(amplitude)
    




len_traj = 1e15
max_lagtime = int(0.25 * len_traj)
msd_nbpt = 500
total_lag_time = np.array(np.unique([int(lag) for lag in np.floor(np.logspace(0, np.log10(max_lagtime), msd_nbpt))]))* p.dt
msd = []



for lag in total_lag_time:
    msd.append(integral_at_t(lag))
msd = np.array(msd)
    
plt.loglog(total_lag_time  * D0 / 100, msd/100)
plt.loglog(total_lag_time  * D0 / 100, (2*D_eff*total_lag_time)/100,color='r',linestyle='--')
plt.xlabel(r'$\frac{Dt}{a^2}$',fontsize=16)
plt.ylabel(r'$\frac{\langle \theta^2 \rangle}{a^2}$',fontsize=10)
plt.title('Analytical integration dt=1e-4,Ampl = 6, red = 2*D_eff*t')
plt.show()



