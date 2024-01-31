from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from class_diffusion_potential import *

T_K = 300
k_b = 1.3806452e-23
R_m = 1e-6
m_kg = 1.1e-14 
viscosity_NS_m2 = 0.001
load = 6*np.pi*viscosity_NS_m2*R_m 
rotational_drag = 8*np.pi*viscosity_NS_m2*R_m**3
tau = m_kg/load
einstein_diffusion = k_b*T_K/load
rotational_einstein_diff = k_b*T_K/rotational_drag
dt_s = 100*tau
frequency = 26
space_step = 1e-12

simulation_f29 = DiffusionSimulation(frequency = 29)
simulation_f28 = DiffusionSimulation(frequency = 28)
simulation_f27 = DiffusionSimulation(frequency = 27)
simulation_f26 = DiffusionSimulation(frequency = 26)
simulation_f25 = DiffusionSimulation(frequency = 25)
simulation_f24 = DiffusionSimulation(frequency = 24)
simulation_f23 = DiffusionSimulation(frequency = 23)

simu_list = [simulation_f29, simulation_f28, simulation_f27, simulation_f26, simulation_f25, simulation_f24, simulation_f23]

for sim in simu_list:

    #A,B = sim.Lifson_Jack_vs_amp_noforce(np.linspace(0, 0.5, 1000))
    A,B = sim.LJ_mean_denominator(np.linspace(0, 5, 1000))
    rot_array = np.ones(len(A))
    rot_array *= rotational_einstein_diff
    plt.plot(A,rot_array,label = 'Simulated diffusion')
    plt.plot(A, B, label=f'periodicity\n ({sim.frequency})')
    plt.xlabel('Barrier amplitude [k_b T]')
    plt.ylabel('D_eff [rad².s-1]')
    plt.legend()
    
plt.show()