# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:12:32 2024

@author: CBS
"""
import numpy as np
from Class_particle_in_potential import *
import matplotlib.pyplot as plt
import time

part = DiffusionSimulation(dt = 1e-3)
N = 10000000
w = part.generate_seq(N)
A = 2
xstart = 0
t0 = time.time()
traj_standard = np.unwrap(part.proceed_traj1(w, N, A))
print(f'run_standard(): Serial done in {time.time() - t0:.5f} s')

t1 = time.time()
traj_test = np.unwrap(part.proceed_traj1(w, N, A))
print(f'test(): Serial done in {time.time() - t1:.5f} s')
t = np.arange(N)
plt.plot(t,traj_standard - traj_test)
plt.title('traj difference modulo(traj)/modulo(x+dx)')
plt.show()
plt.plot(t,traj_standard,label = 'standard')
plt.plot(t,traj_test, label = 'test')
plt.legend()
plt.show()