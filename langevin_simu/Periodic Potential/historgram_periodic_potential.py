from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_v2_periodic_potential import *
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import jv

traj_matrix = []


def get_trajs():
    traj_matrix = []
    trajs = np.load(f'trajectories_1000000,nb_traj_1000points_amplitude_1kT,frequency_10_dt_0.001_torque_0kT.npy')
    [traj_matrix.append(np.mod(traj,2*np.pi)-np.pi) for traj in trajs]                  
    return traj_matrix

def hist_at_t(t,trajs):
    matrix_t = [trajs[i][t]  for i in range(len(trajs))]
    return matrix_t



def histogram_periodic(trajs,t,bins):
    def plot_hist(t,bins):
        plt.hist(hist_at_t(t,trajs), bins=bins, range=(-np.pi, np.pi), density=True, alpha=0.5, label='Histogram')
        
        plt.xlabel('angle [rad]')
        plt.ylabel('density')
    
        # Plot theoretical distribution curve
        space = np.linspace(-np.pi, np.pi, 100)
        plt.plot(space, theory_distrib_i(space), label='Theoretical Distribution', color='red')
        plt.plot(space, Bessel_function(space),linestyle = '--',color = 'black')
        plt.show()
    
    def theory_distrib_i(theta ):
        #center on 0
        GB = p.integrand1(theta+np.pi/2, 1)
        integral = p.factor1(1)
        N = 1/8 #normalisator
        W_theta = GB * N #GB*(N-p.rotational_drag*integral)
        return W_theta
    
    def Bessel_function(x):
        return 2*np.pi*jv(0,10*(x))*1/8
        
    def theo_distrib():
        space = np.linspace(-np.pi, np.pi, 100)
        plt.plot(space, theory_distrib_i(space))
        
    # Assuming DiffusionSimulation2 class definition exists elsewhere in your code
    p = DiffusionSimulation2(dt=0.001, frequency=10, torque=0)
    plot_hist(t, bins=200)
# Example usage:
#\plot_hist(t=999999, bins=200)



# A faire: 10000 realisations à 2kT, trouver comment fixer la densité à 1
    
