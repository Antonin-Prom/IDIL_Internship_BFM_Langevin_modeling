import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from Class_particle_in_potential import Brownian





# Physical parameters
T_K = 300
k_b = 1.3806452e-23
R_m = 1e-6
m_kg = 1.1e-14 
viscosity_NS_m2 = 0.001
load = 6*np.pi*viscosity_NS_m2*R_m 
tau = m_kg/load
diffusion = k_b*T_K/load
d = 1e-12
# Amplitude :
kx = 100*k_b*T_K
energy_argument = kx

# Time step scale:
ratio_phi = load/kx
dt = 100*tau


print(ratio_phi,dt,tau)

def energy(energy_argument,x):
    return energy_argument*np.cos(x*2*np.pi*10e7)

def generate_randseq(N):
    standard_deviation = 1
    # Array of random numbers following a Gaussian distribution with 0 mean and standard deviation of 1  
    W_seq = np.array(np.random.normal(0,standard_deviation,N))
    return W_seq

# Generating the walk in the sinusoidal potential define
def static_process(N,A):
    x = 0
    time = 0
    stored_position = []
    w = generate_randseq(N)
    for i in np.arange(0,N):
        dx = -(1/load)*dt*(energy(A,x+d) - energy(A,x))/d
        dx = dx + w[i]
        x = x + dx
        time = time + dt
        stored_position.append(x)
        dx = 0
    return stored_position

def mean_square_displacement(array):
    msd = []
    centered_array = array - np.mean(array)
    for j in range(1,int(len(centered_array))):
        M = np.mean((centered_array[:-j] - centered_array[j:])**2)
        msd.append(M)
    return np.array(msd)

def msd_W_trajs(W):
    msd_list = []
    for j in range(W):
        traj = static_process(N,kx)
        msd_list.append(mean_square_displacement(traj))
    return np.array(msd_list)

N = 100000
duration = N*dt

t = np.arange(0,N)*dt
time = np.arange(0,N-1)*dt
traj1 = static_process(N,kx)
plt.plot(t,traj1)
plt.show()

msd1 = mean_square_displacement(traj1)

W = 50
msd_array = msd_W_trajs(W)
mean_msd = np.zeros(N-1)
for msd in msd_array:
    mean_msd = np.add(mean_msd,msd)
mean_msd *= 1/W

def linear_fit(t, D):
    return 6 * D * t

# Initial guess for the diffusion coefficient
initial_guess = 1.0

# Perform the fit
params, covariance = curve_fit(linear_fit, time, mean_msd, p0=initial_guess)

# Extract the diffusion coefficient
diffusion_coefficient = params[0]
print("diffusion_coefficient",diffusion_coefficient,diffusion)
# Plot the data and the fit
plt.plot(time, mean_msd, label='MSD Data')
plt.plot(time, linear_fit(time, diffusion_coefficient), label='Linear Fit')
plt.xlabel('Time')
plt.ylabel('MSD')
plt.legend()
plt.show()




"""
x = t_1
y = mean_msd
coefficients = np.polyfit(x, y, 1)
slope, intercept = coefficients

# Create a linear function using the obtained coefficients
linear_fit = np.poly1d(coefficients)

# Generate y values for the linear fit line
fit_line = linear_fit(x)

# Plot the original data and the linear fit line
plt.scatter(x, y, label='Original Data')
plt.plot(x, fit_line, color='red', label='Linear Fit')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear Fit Example')
plt.legend()

# Show the plot
plt.show()

# Display the slope and intercept
print(f"Slope: {slope}, Intercept: {intercept}")
 """