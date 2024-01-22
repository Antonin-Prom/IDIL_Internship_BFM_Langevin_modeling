from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from class_diffusion_potential import *

""" 
Simulation of rotational diffusion in periodic potential with torque
+
Lifson jackson with force 
+
Courbe de tendance --> faire distance entre courbe de tendance et LJ --> déduire l'amplitude ou la fréquence 
"""

simulation_f = DiffusionSimulation(frequency = 26, torque = 16) #kbT

def compare_LJ(amplitude):
    #sim_coef = simulation_f26.simulate(amplitude, nb_traj, traj_length)
    LJ_noforce = simulation_f.lifson_jackson_noforce(amplitude)
    LJ_force = simulation_f.lifson_jackson_force(amplitude)
    return LJ_noforce,LJ_force

def compare_array(amplitude_array):
    LJ_noforce_array,LJ_force_array = np.zeros(len(amplitude_array)),np.zeros(len(amplitude_array))
    counter = 0
    for amplitude in amplitude_array:
        LJ_noforce_array[counter],LJ_force_array[counter] = compare_LJ(amplitude)
        counter += 1
    return LJ_noforce_array,LJ_force_array 

def compare_with_period():
    simulation_f20 = DiffusionSimulation(frequency = 20)
    simu_list = [ simulation_f20]
    for sim in simu_list:
        A,B = sim.Lifson_Jack_vs_amp_noforce(np.linspace(0, 5, 1000))
        plt.plot(A, B, label=f'LJ no torque 20 periods')

def plot_LJs():
    amplitude_array = np.linspace(0, 5, 1000)
    LJ_noforce,LJ_force = compare_array(amplitude_array)
    compare_with_period()
    plt.plot(amplitude_array, LJ_noforce, label= 'LJ no torque 26 periods')
    plt.plot(amplitude_array, LJ_force, label= 'LJ torque = 15kT 26 periods' )
    plt.xlabel('Barrier amplitude [k_b T]')
    plt.ylabel('D_eff [rad².s-1]')
    plt.legend()

plot_LJs()
plt.show()
