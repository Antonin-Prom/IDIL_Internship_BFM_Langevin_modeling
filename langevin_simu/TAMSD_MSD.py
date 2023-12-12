import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter


#finite spatial step
d = 1e-9

# timesteps
dt = 0.001
timesteps = 5000
fig, (ax1,ax4,ax3) = plt.subplots(3, 1)

#physical parameters
kT = 1.38e-23*300
R = 1e-6
eta = 1e-3
friction = 6*np.pi*eta*R
diffusion = kT/friction
sd = np.sqrt(2*diffusion*dt)

a = 1e7 #N.m^-3
b = 1e-6 #N.m^-1

# Quadratic Potential
def energy(A,x,r):
    return 0

"""
def energy(A,x,r):
    return A*sin(r*x)
"""

amplitude = 3
freq = 1/(d*10000)
stored_position = []
stored_time = []
stored_potential = []

x = 0
time = 0.0 


def mean_square_displacement(positions):
    position = np.array(positions)
    mean_array = []
    # Each iteration increase the difference
    for j in range(1,int(len(position)/2)):
        # Store the differences 
        diff_array = position[:-j] - position[j:]
        M = np.mean(diff_array*diff_array)
        mean_array.append(M)
    #Convert into a set of delta t
    time_array = np.arange(1,int(len(position)/2))*dt
    return mean_array, time_array

def time_averaged_MSD(positions):
    total_trajectory = len(positions)
    x = positions
    dtau = 1
    max_lagtime = int(len(x)/2)
    time_array = np.arange(1,int(len(x)/2))*dt
    TAMSD = []
    #iterating over lag_time
    for n in range(1,max_lagtime): 
        #TAMSD for a given lag time n*dtau 
        M = total_trajectory  
        ntamsd = 0
        for m in range(M-n):
            ntamsd += ((x[(n+m)*dtau] - x[m*dtau])**2)/(M-n)
        TAMSD.append(ntamsd)
    return TAMSD,time_array


for i in np.arange(0,timesteps):
    dx = -(1/friction)*dt*(energy(amplitude, x+d,freq) - energy(amplitude, x,freq))
    dx = dx + np.random.normal(0.0,sd)

    x = x + dx
    time = time + dt
    potential = energy(amplitude, x,2)

    stored_position.append(x)
    stored_time.append(time)
    stored_potential.append(potential)
    dx = 0



ytic = np.linspace(-1e-5, 1e-5, 11)
ax1.set_yticks(ytic)
ax1.set_yticklabels([f'{val*1e6:.1f}' for val in ytic])  # Display in micrometers
ax1.set_ylim(-1e-5, 1e-5)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Position (μm)')  # Update ylabel to micrometers
ax1.set_title("Particle position in a potential well")
ax1.plot(stored_time, stored_position, color='navy')



mean_displacement_array,delta_t = mean_square_displacement(stored_position)
y4tic = np.linspace(-10e-12,10e-12,31)
ax4.set_yticks(y4tic)
ax4.set_yticklabels([f'{val*1e12:.0f}' for val in y4tic]) 
ax4.plot(delta_t,mean_displacement_array)
ax4.set_xlabel('Lag time $\Delta$ t (ms)')
ax4.set_ylabel('Mean squared displacement (μm²)')
ax4.set_title("Mean Square Displacement")

tamsd,lag_time = time_averaged_MSD(stored_position)
ax3.plot(lag_time,tamsd)
ax3.set_title('Time averaged mean squared displacement')
ax3.set_xlabel('Lag time $\Delta$ t (ms)')
ax3.set_ylabel('Time Averaged Mean squared displacement ')

print('Difference between MSD and TAMSD' , np.mean(np.array(mean_displacement_array)),np.array(tamsd))
plt.show()

