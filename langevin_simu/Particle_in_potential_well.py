import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter


#finite spatial step
d = 1e-12

# timesteps
dt = 1e-7
timesteps = 100000

fig, ((ax1,ax4),(ax3,ax2)) = plt.subplots(2, 2)

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
    return A*(x)**2 

stiffness = 1e-3
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


for i in np.arange(0,timesteps):
    dx = -(1/friction)*dt*(energy(stiffness, x+d,freq) - energy(stiffness, x,freq))
    dx = dx + np.random.normal(0.0,sd)

    x = x + dx
    time = time + dt
    potential = energy(stiffness, x,2)

    stored_position.append(x)
    stored_time.append(time)

    stored_potential.append(potential)
    dx = 0

def find_interval(j,int_list): # The list must be sorted min to max
    for i in range(1,len(int_list)):
        if int_list[i-1] <= j < int_list[i] :
            return i
        else :
            continue

def histogram(pos_array,N):
    interval_list = np.linspace(min(pos_array),max(pos_array),N)
    """ The goal is to produce an histogramm with in height the number of times where the particle dwelled, depending of the position"""
    #Height value :
    bar_height = np.zeros(N)
    """ Iterate through the position array and update the number of times that the particle dwelled for each intervall """
    for x in pos_array :
        index = find_interval(x,interval_list)
        bar_height[index] += 1
    normalizer = sum(bar_height)
    bar_height = bar_height/normalizer
    return [interval_list,bar_height]



""" 
The first displayed interval should begin at the value of 1/100 of the maximal height to avoid a set of empty interval.
The x_axis must be translated in positions, and display the coordinates of the intervals.
The interval 0 begin at the min position
"""      


def produce_bar(N):

    hist = histogram(stored_position,N)
    """ max_value = max(hist[1])
    min_value = 0.01*max_value

    min_index = find_interval(min_value,hist[1]) - 1 # -1 Because the function return the superior interval
 
    updated_hist = hist[1][min_index:]
    updated_N = len(updated_hist)
    bar_number = np.arange(0,updated_N) """
    updated_hist = hist[1]
    bar_number = np.arange(0,N)
    return bar_number,updated_hist,hist[0]

N = 20

bar_number,updated_hist,new_xaxis = produce_bar(N)
ax3.bar(bar_number, updated_hist)
ax3.set_ylim(0,1.2*max(updated_hist))
ax3.set_xticks(bar_number)  
scaled_xaxis = [x * 1e6 for x in new_xaxis]
# Format and set the tick labels
formatted_labels = ['{:.2}'.format(x) for x in scaled_xaxis]
ax3.set_xticklabels(formatted_labels[::len(formatted_labels)//len(bar_number)],rotation=45)
ax3.set_xlabel('Position (μm)')
ax3.set_ylabel('Probability ')
ax3.set_title("Probability distribution of position")

ytic = np.linspace(-1e-5, 1e-5, 11)

ax1.set_yticks(ytic)
ax1.set_yticklabels([f'{val*1e6:.1f}' for val in ytic])  # Display in micrometers
ax1.set_ylim(-1e-5, 1e-5)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Position (μm)')  # Update ylabel to micrometers
ax1.set_title("Particle position in a potential well")
ax1.plot(stored_time, stored_position, color='navy')


X = np.arange(-10e-5,10e-5,1e-7)
x2tic = np.linspace(-10e-5,10e-5,21)
ax2.set_xticks(x2tic)
ax2.set_xticklabels([f'{val*1e6:.0f}' for val in x2tic]) 
ax2.set_xlim(-10e-5,10e-5)
ax2.plot(X,energy(stiffness,X,freq))
ax2.set_xlabel('Position (μm)')
ax2.set_ylabel('Energy (u.a)')
ax2.set_title('Energy Potential')

mean_displacement_array,delta_t = mean_square_displacement(stored_position)
y4tic = np.linspace(-10e-12,10e-12,31)
ax4.set_yticks(y4tic)
ax4.set_yticklabels([f'{val*1e12:.0f}' for val in y4tic]) 
ax4.plot(delta_t,mean_displacement_array)
ax4.set_xlabel('Lag time $\Delta$ t (ms)')
ax4.set_ylabel('Mean squared displacement (μm²)')
ax4.set_title("Mean Square Displacement")

plt.show()
