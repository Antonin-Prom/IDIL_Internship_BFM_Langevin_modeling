# %% [markdown]
# This notebook aims to reproduce and compare the results from the publication "Simulation of a Brownian particle in an optical trap" (2013, G.Volpe and G.Volpe). The same physical parameters and numerical values will be used.
# 

# %% [markdown]
# I. White noise
# 1. The white noise function is described by two properties :
# $$ <W(t)> = 0 $$ 
# $$ <W(t)²> = 1 $$
# $$ <W(t_1)W(t_2)> = 0  

# %% [markdown]
# 2. Random Walk :
# The position of a particle in a free random walk is described by :
# $$ \dot x = W(t) \space (1)$$ 
# In a finite difference approach we describe W(t) as a discrete sequence of random values $W_i$ with 0 mean.
# In order to have a variance of $W_i$ of $\frac{1}{\Delta t}$ we set :
# $$ \frac{<(W_i \Delta t )²>}{\Delta t} = 1 \space (2)$$
# In Python we can generate the random sequence of $w_i$ using the python gaussian function and setting the mean to 0 and variance to 1.
# The equation (1) can be written in the finite difference as :
# $$ \frac {x_i - x_{i-1}}{\Delta t} = \frac{w_i}{\sqrt{\Delta t}} \space(3)$$
# $$ x_i = x_{i-1} + w_i \sqrt{\Delta t} \space(4)$$
# 
# Examples of free diffusing trajectories using different values of $\Delta t$

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# %%
# Generating the random sequence

def generate_seq(dt,N):
    standard_deviation = 1  #(1/(2*np.sqrt(dt)))
    W_seq = np.array(np.random.normal(0,standard_deviation,N))
    return W_seq

print(np.std(generate_seq(0.5,500000)),1)

# %% [markdown]
# Comparing the standard deviation over 5e5 random gaussian number with the actual sd in entry of the function.

# %%
# Proceed Finite difference

def walk(duration,dt):
    N = int(duration/dt) # Number of step
    x = 0
    positions = []
    W_seq = np.array(generate_seq(dt,N))
    for i in range(N):
        positions.append(x)
        x += W_seq[i]*np.sqrt(dt)
    return np.array(positions)
    

# %%
# Plot:
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

# %% [markdown]
# Now we will compute the standard deviation by averaging the standard deviation over 10000 trajectories
# $$ V = <x²> - <x>² $$

# %%
def mean_sd(nb_ite,dt):
    duration = 30
    standard_deviation = np.zeros(int(duration/dt))
    for i in range(nb_ite):
        # generate for each iteration a random_walk 
        trajectory = walk(duration,dt)
        for j in range(len(trajectory)):
            # computing the standard deviation for each index of each trajectory
            standard_deviation[j] += np.std(trajectory[:j+1])
    standard_deviation  = (standard_deviation/nb_ite)
    return standard_deviation

# %%
# Plot
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15, 4))
sd1 = mean_sd(1000,1.0)
t1 = np.arange(0,30,1)
ax1.plot(t1,sd1)
ax1.set_xlabel('t(s)')
ax1.set_ylabel('x')
ax1.set_title('$\Delta$ t = 1.0')

sd2 = mean_sd(1000,0.5)
t2 = np.arange(0,30,0.5)
ax2.plot(t2,sd2)
ax2.set_xlabel('t(s)')
ax2.set_ylabel('x')
ax2.set_title('$\Delta$ t = 0.5')

sd3 = mean_sd(1000,0.1)
t3 = np.arange(0,30,0.1)
ax3.plot(t3,sd3)
ax3.set_xlabel('t(s)')
ax3.set_ylabel('x')
ax3.set_title('$\Delta$ t = 0.1')
plt.show()

# %%
def mean_trajectory(bundle):
    mean_traj = np.zeros(len(bundle[0]))
    for traj in bundle:
        mean_traj = np.add(mean_traj,traj)
    mean_traj *= 1/len(bundle)
    return mean_traj

def var_mean_traj(mean_traj):
    var_mean = np.zeros(len(mean_traj))
    for i in range(len(var_mean)):
        var_mean[i] = np.std(mean_traj[:i])
    return var_mean

# %%
    

def plot_traj(dt):
    trajectories = []
    for j in range(1000):
        trajectories.append(walk(30,dt))
    plt.figure(figsize=(10, 6))
    #for i in range(3):
    #    plt.plot(np.arange(0,30,dt),trajectories[i]) 
    mean_traj = mean_trajectory(trajectories)
    var_mean = var_mean_traj(mean_traj)
    plt.plot(np.arange(0,30,dt),mean_traj,label='mean_traj')
    plt.plot(np.arange(0,30,dt),var_mean,label='var')

    plt.title('Brownian Motion Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Position')
    plt.legend()
    plt.show()



# %% [markdown]
# Averaging many trajectories and than taking the variance of it seems to be the wrong way of proceeding because we set the mean of our random sequence at 0 (definition of a Brownian walker). So as expected the variance seems pretty low. I tryed anyway because the formulation in the publication was unclear : 
# 
# "which represent the variance around the mean
# position of the freely diffusing random walker obtained by
# averaging over 10,000 trajectories, are roughly the same, independent of Dt. (The small differences are due to the finite
# number of trajectories used in the averaging.)"
# 
# And :
# 
# "their statistical properties do not change, as can be
# seen by comparing the shaded areas, which show the regions within one standard deviation of the mean of 10,000 realizations."
# 

# %%
plot_traj(1)
plot_traj(0.5)
plot_traj(0.1)


