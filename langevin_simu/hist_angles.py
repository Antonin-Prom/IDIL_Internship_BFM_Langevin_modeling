from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_in_potential import *

A = 2
part = DiffusionSimulation(dt = 1e-6,torque = 20)
traj = part.proceed_traj1(10000000,A)
plt.hist(traj,bins = 1000,density=True)
plt.xlabel('angle [rad]')
plt.ylabel('density')
plt.show()

