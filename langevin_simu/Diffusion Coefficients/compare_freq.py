from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from class_diffusion_potential import *

simulation_f29 = DiffusionSimulation(frequency = 29)
simulation_f28 = DiffusionSimulation(frequency = 28)
simulation_f27 = DiffusionSimulation(frequency = 27)
simulation_f26 = DiffusionSimulation(frequency = 26)
simulation_f25 = DiffusionSimulation(frequency = 25)
simulation_f24 = DiffusionSimulation(frequency = 24)
simulation_f23 = DiffusionSimulation(frequency = 23)

simu_list = [simulation_f29, simulation_f28, simulation_f27, simulation_f26, simulation_f25, simulation_f24, simulation_f23]

for sim in simu_list:
    A,B, = sim.Lifson_Jack_Simu()
    plt.plot(A,B,label = str(sim.frequency))
    plt.legend()
    
plt.show()