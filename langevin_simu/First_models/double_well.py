import numpy as np
import matplotlib.pyplot as plt
import time

#finite spatial step
d = 1e-9

# timesteps
dt = 0.001
timesteps = 200000


#physical parameters
kT = 1.38e-23*300
R = 1e-6
eta = 1e-3
friction = 6*np.pi*eta*R
diffusion = kT/friction
sd = np.sqrt(2*diffusion*dt)

a = 1e7 #N.m^-3
b = 1e-6 #N.m^-1

def energy(a,b,x):
    return a*x**4/4 - b*x**2/2


stored_position = []
stored_time = []
stored_potential = []

x = 0
time = 0.0 
for i in np.arange(0,timesteps):
    dx = -(1/friction)*dt*((energy(a,b, x+d) - energy( a,b, x))/d)
    dx = dx + np.random.normal(0.0,sd)

    x = x + dx
    time = time + dt
    potential = energy( a,b,x)

    stored_position.append(x)
    stored_time.append(time)
    stored_potential.append(potential)
    dx = 0

plt.xlabel('t[s]')
plt.ylabel('x[m]')
plt.title("Dynamic transition between two equilibrium state")
plt.plot(stored_time,stored_position,color='royalblue')
plt.show()