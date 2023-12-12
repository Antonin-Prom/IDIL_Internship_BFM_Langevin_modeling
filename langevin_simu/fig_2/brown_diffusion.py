import numpy as np
import matplotlib.pyplot as plt


# Physical parameters for a micro bead in water at R.T

T_K = 300
k_b = 1.3806452e-23
R_m = 1e-6
m_kg = 1.1e-14 
viscosity_NS_m2 = 0.001
load = 6*np.pi*viscosity_NS_m2*R_m 
tau = m_kg/load
diffusion = k_b*T_K/load

# Simulation parameters
dt_s = 10e-9  




def generate_randseq(N):
    standard_deviation = 1
    # Array of random numbers following a Gaussian distribution with 0 mean and standard deviation of 1  
    W_seq = np.array(np.random.normal(0,standard_deviation,N))
    return W_seq

def walk_diffusion(N,w):
    x = np.zeros(N)
    for i in range(1,N):
        x[i] = x[i-1] + np.sqrt(2*diffusion*dt_s)*w[i]
    return x

def walk_inertia(N,w):
    x = np.zeros(N)
    # As a term in [i-2] is needed to compute x[i] we must define x[1] first.
    x[1] = ((2+dt_s*(1/tau))/(1+dt_s*(1/tau)))*x[0] + (np.sqrt(2*k_b*T_K*load)/(m_kg*(1 + dt_s*(1/tau))))*dt_s**(3/2)*w[1]
    for i in range(2,N):
        x[i] = ((2+dt_s*(1/tau))/(1+dt_s*(1/tau)))*x[i-1] - (1/(1+dt_s*(1/tau)))*x[i-2] + (np.sqrt(2*k_b*T_K*load)/(m_kg*(1 + dt_s*(1/tau))))*dt_s**(3/2)*w[i]
    return x

# Comparing 2 trajectories with and with out inertia


duration1 = tau
duration2 = 100*tau
N1 = int(duration1/dt_s)
print(N1)
N2 = int(duration2/dt_s)
print(N2)

w = generate_randseq(N2)

fig,(ax1,ax2) = plt.subplots(2,1)

t1 = np.linspace(0,duration1,N1)
ax1.plot(t1,walk_inertia(N1,w),color='red', label='inertial')
ax1.plot(t1,walk_diffusion(N1,w),color='black', linestyle='dotted', label='non-inertial')
ax1.set_xlabel('t/$\tau$')
ax1.set_ylabel('x(m)')
ax1.legend()

t2 = np.linspace(0,duration2,N2)
ax2.plot(t2,walk_inertia(N2,w),color='red', label='inertial')
ax2.plot(t2,walk_diffusion(N2,w),color='black', linestyle='dotted', label='non-inertial')
ax2.set_xlabel('t/tau')
ax2.set_ylabel('x(m)')
ax2.legend()
plt.show()


def auto_correlation():
    
    duration = 3*tau
    time_points = np.arange(0,duration,dt_s)
    N = len(time_points)
    w = generate_randseq(N)
    position = walk_inertia(N,w)
    velocities = np.diff(position, axis=0)
    centered_velocities = velocities - np.mean(velocities)

    autocorr = np.convolve(centered_velocities, centered_velocities[::-1], mode='full')
    autocorr /= autocorr[len(autocorr)//2]  # Normalize by autocorrelation at zero lag
    time_array = np.arange(int(len(autocorr)))*dt_s/tau
    return time_array,autocorr


x,C = auto_correlation()
plt.xlabel('t/tau')
plt.ylabel('Cv(t),[a.u]')
plt.plot(x,C)
plt.show()

