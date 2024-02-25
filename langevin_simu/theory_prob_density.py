# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:07:16 2024

@author: CBS
"""
from scipy.integrate import quad
import numpy as np
from Class_particle_in_potential import *
import matplotlib.pyplot as plt

simu = DiffusionSimulation(dt=1e-6)
X = np.linspace(0,2*np.pi,1000)

A = 2
def F(x):
    return np.exp(-simu.tilted_periodic_potential(A, x))

def prob_density(X,A):
    W = np.zeros(len(X))
    counter = 0
    for x in X :
        damping = simu.rotational_drag
        G = simu.tilted_periodic_potential(A, x)
        integration_result, _ = quad(F, 0, x)
        W[counter] = np.exp(-G)*(1 - damping * integration_result)
        counter += 1
    return W

W = prob_density(X,A)
plt.plot(X,W)
plt.xlabel('angle')
plt.ylabel('Probability Density')
plt.show()