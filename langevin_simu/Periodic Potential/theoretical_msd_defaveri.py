from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_v2_periodic_potential import *
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import jv

p = DiffusionSimulation2(dt=0.001, frequency=10, torque=0)
amplitude = 3
D_eff = p.lifson_jackson_noforce(amplitude)
D0 = p.rotational_einstein_diff

def integral_at_t(t):
    
    def integrand(y, amplitude):
        return y**2 * p.integrand1(y * np.sqrt(t), amplitude) * np.exp(-(y**2) / (4 * D_eff)) / (4 * np.pi * D0)
    
    low_lim = -np.pi / np.sqrt(t)
    high_lim = np.pi / np.sqrt(t)
    
    angle_space = np.linspace(-np.pi, np.pi, 1000)
    y = angle_space / np.sqrt(t)
    
    def results(amplitude):  
        result, _ = quad(integrand, low_lim, high_lim, args=(amplitude,))
        return result
    
    return t * results(amplitude)
    
len_traj = 1e6
max_lagtime = int(0.25 * len_traj)
msd_nbpt = 1000
total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, np.log10(max_lagtime), msd_nbpt))])
msd = []



for lag in total_lag_time:
    msd.append(integral_at_t(lag))
    
plt.plot(total_lag_time * p.dt_s, msd)
plt.show()
