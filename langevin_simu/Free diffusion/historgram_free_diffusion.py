from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/langevin_simu/')
from Class_particle_v2_free_diffusion import *
from scipy.stats import norm
from scipy.optimize import curve_fit


traj_matrix = []
def generate_mean_traj(N,L,A,torque):
    part = DiffusionSimulation(dt = 1e-5,torque = 100)
    for i in range(N):
        traj = part.proceed_traj1(L ,A)
        traj_matrix.append(traj)
    mean_traj = np.mean(traj_matrix,axis=0)
    return mean_traj

#mean_traj = generate_mean_traj(1,1000000,3,100)


def get_trajs(amplitude,dt):
    trajs = np.load(f'trajectories_100000points_amplitude_0kT_dt_1e-05_torque_0.npy')                  
    return trajs

def hist_at_t(amplitude,dt,t):
    trajs = get_trajs(amplitude,dt)
    matrix_t = [trajs[i][t]  for i in range(len(trajs))]
    return matrix_t


def plot_hist(amplitude,dt,i):
    trajs = get_trajs(amplitude,dt)
    traj = trajs[i]
    plt.hist(traj,bins = 2600,density=True)
    plt.xlabel('angle [rad]')
    plt.ylabel('density')
    plt.show()

def normal_theory(mu,t,x):
    part = DiffusionSimulation2()
    D = part.rotational_einstein_diff
    t = t*1e-5
    p_theory = (1/np.sqrt(4*np.pi*D*t))*np.exp(- ((x - mu)**2)/(4*D*t))
    return p_theory

def normal_longtime_correction(mu,t,x):
    part = DiffusionSimulation2()
    D = part.rotational_einstein_diff
    y = part.rotational_drag
    t = t*1e-5
    p_theory = np.sqrt(y/(2*np.pi*D))*np.exp(- (y*x**2)/(2*D))
    return p_theory

def normal_theory_full_correction(mu,t,x):
    part = DiffusionSimulation2(dt=1e-5)
    D = part.rotational_einstein_diff
    y = part.rotational_drag*1000
    t = t*1e-5
    interm = y/(2*D*(1-np.exp(-2*y*t)))
    p_theory = np.sqrt(y/(2*np.pi*D*(1-np.exp(-2*y*t))))*np.exp(-interm*x**2)
    return p_theory

"""                                  
t = 100
matrix_at_t = hist_at_t(0,1e-5,t)

plt.hist(matrix_at_t,bins = 1000,density=True)
(mu, sigma) = norm.fit(matrix_at_t)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')    
    
plt.plot(x,normal_theory(mu,t,x), linestyle='--', color='red', label='Theoretical Normal Distribution')

plt.title(f'Histogram of positions at t={t*1e-5:.3f}s over 10000 realisations')
plt.xlabel('angle [rad]')
plt.ylabel('density')
plt.legend()
plt.show()

"""

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

times = [0, 100, 1000, 10000]  # Adjust time values as needed
for i, t in enumerate(times):
    matrix_at_t = hist_at_t(0, 1e-5, t)
    ax = axs[i//2, i%2]  # Get the current axis
    ax.hist(matrix_at_t, bins=100, density=True)
    (mu, sigma) = norm.fit(matrix_at_t)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
    #ax.plot(x, normal_theory(mu, t, x), linestyle='--', color='red', label='Theoretical Normal Distribution')
    ax.plot(x, normal_theory_full_correction(mu, t, x), linestyle='--', color='green', label='Corrected Normal Distribution')
    ax.plot(x, normal_longtime_correction(mu, t, x), linestyle='--', color='orange', label='normal_longtime_correction Distribution')
    ax.set_title(f'Histogram at t={t*1e-5:.3f}s over 10000 realisations')
    ax.set_xlabel('angle [rad]')
    ax.set_ylabel('density')
    ax.legend()

plt.tight_layout()
plt.show()