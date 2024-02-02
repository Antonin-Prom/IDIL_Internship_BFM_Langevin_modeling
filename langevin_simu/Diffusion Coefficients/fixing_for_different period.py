from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from class_diffusion_potential import *

simulation_f29 = DiffusionSimulation(frequency = 2)
simulation_f28 = DiffusionSimulation(frequency = 10)
simulation_f27 = DiffusionSimulation(frequency = 20)


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


simu_list = [simulation_f29, simulation_f28, simulation_f27]

amplitude_array = (np.linspace(0, 4, 100))
rot_array = np.ones(len(amplitude_array))
rot_array *= rotational_einstein_diff
for sim in simu_list:
    coef_array = np.zeros(len(amplitude_array))
    counter = 0
    for amp in amplitude_array:
        coef_array[counter] = sim.simulate(amp,100,4000)
        counter += 1
    plt.plot(amplitude_array,coef_array,label=f'periodicity\n ({sim.frequency})')
plt.plot(amplitude_array,rot_array,label = 'einstein_diffusion',color = 'black')
plt.show()