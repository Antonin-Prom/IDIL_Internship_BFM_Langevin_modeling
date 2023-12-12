import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Generating the random sequence W
def generate_seq(dt,N):
    standard_deviation = 1
    # Array of random numbers following a Gaussian distribution with 0 mean and standard deviation of 1  
    W_seq = np.array(np.random.normal(0,standard_deviation,N))
    return W_seq


def walk(duration,dt):
    N = int(duration/dt) # Number of step
    W_seq = np.array(generate_seq(dt,N))
    x = np.zeros(N)
    for i in range(1,N):
        #Rescaling the sequence in order to have the right variance
        x[i] = x[i-1] + W_seq[i]*np.sqrt(dt)
    return x # Returning an array containing the walker positions
    


# Ploting the brownian trajectory:
""" 
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15, 4))

t1 = np.arange(0,30,1)
x1 = walk(30,1.0)
ax1.scatter(t1,x1,s=1)

t2 = np.arange(0,30,0.5)
x2 = walk(30,0.5)
ax2.scatter(t2,x2,s=1)

t3 = np.arange(0,30,0.1)
x3 = walk(30,0.1)
ax3.scatter(t3,x3,s=1)

plt.show()
"""

def mean_sd(nb_ite,dt):
    duration = 30
    nb_tstep = int(duration/dt)
    standard_deviation = np.zeros(nb_tstep)
    mean_traj = np.zeros(nb_tstep)
    matrix = []
    for i in range(nb_ite):
        # generate for each iteration a random_walk 
        trajectory = walk(duration,dt)
        matrix.append(trajectory)
    matrix = np.array(matrix)
    for j in range(nb_tstep):
        # for each column (time_step) mean and sd
        mean_traj[j] = np.mean(matrix[:,j])
        standard_deviation[j] += np.std(matrix[:,j])
        total = np.array(mean_traj) + np.array(standard_deviation)
    return total


# Ploting the standard deviations
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15, 4))
sd1 = mean_sd(10000,1.0)
t1 = np.arange(0,30,1)
#x1 = walk(30,1.0)
#ax1.plot(t1,x1)
ax1.fill_between(t1, sd1, -sd1, color='gray', alpha=0.5, label='Shaded Area')
ax1.set_xlabel('t(s)')
ax1.set_ylabel('x')
ax1.set_title('$\Delta$ t = 1.0')

sd2 = mean_sd(10000,0.5)
t2 = np.arange(0,30,0.5)
#x2 = walk(30,0.5)
#ax2.plot(t2,x2)
ax2.fill_between(t2, sd2, -sd2, color='gray', alpha=0.5, label='Shaded Area')
ax2.set_xlabel('t(s)')
ax2.set_ylabel('x')
ax2.set_title('$\Delta$ t = 0.5')

sd3 = mean_sd(10000,0.1)
t3 = np.arange(0,30,0.1)
ax3.fill_between(t3, sd3, -sd3, color='gray', alpha=0.5, label='Shaded Area')
#x3 = walk(30,0.1)
#ax3.plot(t3,x3)
ax3.set_xlabel('t(s)')
ax3.set_ylabel('x')
ax3.set_title('$\Delta$ t = 0.1')
plt.show()
