import numpy as np
import matplotlib.pyplot as plt
import math
import random
from tkinter import *
from array import *
import time
import datetime


# Initialisation of potential :
amplitude = 3
k = 2

#intial position
x = 0

#Physical parameters!
kT = 1
friction = 1
diffusion = kT/friction

#finite spatial step
d = 0.001

# timesteps
dt = 0.001
timesteps = 10000

#standard deviation
sd = np.sqrt(2*diffusion*dt)

random.seed(151)
x=0
time = 0.0


stored_position = []
stored_time = []
stored_potential = []
stored_mean_square = []

def energy(A,x,k):
    return A*np.cos(k*x)

def mean_square(array):
    #For an initial postion at 0 
    #the mean square displacement is the sum of all (actual_position - ref_position)² divided by all the position
    M = 0
    mean_square = 0
    for m in array:
        M += m
    for i in array:
        mean_square += (i*i)
    return mean_square/m

def mean_square_array(array):
    for j in range(len(array)):
        stored_mean_square.append(mean_square(array[0:j+1]))
    return stored_mean_square


for i in np.arange(0,timesteps):
    const_torque = 0.0008
    dx = -(1/friction)*(dt*(energy(amplitude, x+d, k) - energy( amplitude,x, k)) + const_torque)
    
    #add gaussian noise
    dx += np.random.normal(0.0,sd)
    
    #add torque:
    
    
    #update x
    x = x + dx
    
    #update time
    time = time + dt

    potential = energy( amplitude,x, k)
    stored_position.append(x)
    stored_time.append(time)
    stored_potential.append(potential)
    dx = 0

stored_mean_square = mean_square_array(stored_position)

plt.figure()
plt.subplot(222)
plt.plot(stored_time, stored_position)
plt.xlabel('time[s]')
plt.ylabel('x[nm]')

plt.subplot(221)
plt.plot(stored_time, stored_mean_square)
plt.xlabel('time[s]')
plt.ylabel('<x²>[nm]²')


# plt.xlabel('time')
# plt.ylabel('position')
# plt.xlim(0,10)

# plt.plot(stored_time,stored_position,stored_position,color='navy')
plt.show()