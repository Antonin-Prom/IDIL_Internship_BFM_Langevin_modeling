import numpy as np
T_K = 300
k_b = 1.3806452e-23
R_m = 1e-6
m_kg = 1.1e-14 
viscosity_NS_m2 = 0.001
load = 6*np.pi*viscosity_NS_m2*R_m 
tau = m_kg/load
diffusion = k_b*T_K/load
print("diffusion", diffusion)

dt = 100*tau
N = 100000
duration = N*dt
slope = 13332.0270463551
exp_diff = 2857/(6*1000)
print("experimental_diffusion", exp_diff)