import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad



T_K = 300
k_b = 1.3806452e-23
R_m = 1e-6
m_kg = 1.1e-14 
viscosity_NS_m2 = 0.001
load = 6*np.pi*viscosity_NS_m2*R_m 
rotational_drag = 8*np.pi*viscosity_NS_m2*R_m**3
tau = m_kg/load
einstein_diffusion = k_b*T_K/load
rotational_einstein_diff = k_b*T_K/rotational_drag
dt_s = 100*tau
frequency = 26
space_step = 1e-12

def generate_seq(N):
    standard_deviation = 1 
    W_seq = np.array(np.random.normal(0,standard_deviation,N))
    return W_seq

def mean_square_displacement(array):
    msd = []
    centered_array = array - np.mean(array)
    for j in range(1,int(len(centered_array))):
        M = np.mean((centered_array[:-j] - centered_array[j:])**2)
        msd.append(M)
    return np.array(msd)

def periodic_potential(A,x):
    return A*np.cos(x*frequency)

def static_process(N,A):
    x = 0
    stored_position = []
    w = generate_seq(N)
    for i in np.arange(0,N):
        dx = -(1/rotational_drag)*dt_s*(periodic_potential(A,x+space_step) - periodic_potential(A,x))/space_step
        dx = dx + np.sqrt(2*rotational_einstein_diff*dt_s)*w[i]
        x = np.mod(x + dx, 2*np.pi)
        stored_position.append(x)
        dx = 0
    return stored_position

def msd_W_trajs(W,N,amplitude):
    msd_list = []
    mean_msd = np.zeros(N-1)
    for j in range(W):
        traj = np.unwrap(static_process(N,amplitude*k_b*T_K))
        msd_list.append(mean_square_displacement(traj))
    for msd in msd_list:
        mean_msd = np.add(mean_msd,msd)
    mean_msd *= 1/W
    mean_msd = mean_msd[1000:-int(len(mean_msd)/2)]
    return mean_msd


def slope_fit(t,mean_msd):
    slope, _ = np.polyfit(t, mean_msd, 1)
    simulated_diffusion = slope/2
    return simulated_diffusion

def factor1(amplitude):  
    result, _ = quad(integrand1, -np.pi/26, np.pi/26, args=(amplitude))
    return result
def integrand1(x,amplitude):
    return np.exp(periodic_potential(amplitude,x))

def lifson_jackson(amplitude):
    lifson_jackson1 = rotational_einstein_diff*(2*np.pi/frequency)**2/((factor1(amplitude))*(factor1(-amplitude)))
    return lifson_jackson1 

def Bessel(A):
    I0 = np.exp(A)/(np.sqrt(2*np.pi*A)) 
    D = rotational_einstein_diff/(I0*I0)
    return D

def simulate(amplitude,nb_traj,traj_length):
    mean_msd = msd_W_trajs(nb_traj,traj_length,amplitude)
    t = np.arange(len(mean_msd))*dt_s
    simulated_diffusion = slope_fit(t,mean_msd)
    Lifson_jackson_diffusion_coefficient = lifson_jackson(amplitude)
    bessel_diffusion_coefficient = Bessel(amplitude)
    return simulated_diffusion,Lifson_jackson_diffusion_coefficient,bessel_diffusion_coefficient

def compare():
    amplitude_array = np.linspace(0,3,100)
    simulated_diffusion_array,Lifson_jackson_diffusion_coefficient_array,bessel_diffusion_coefficient_array, = np.zeros(len(amplitude_array)),np.zeros(len(amplitude_array)),np.zeros(len(amplitude_array))
    counter = 0
    for amplitude in amplitude_array:
        simulated_diffusion,Lifson_jackson_diffusion_coefficient,bessel_diffusion_coefficient = simulate(amplitude,nb_traj = 100,traj_length=5000)
        simulated_diffusion_array[counter] = simulated_diffusion
        Lifson_jackson_diffusion_coefficient_array[counter] = Lifson_jackson_diffusion_coefficient
        bessel_diffusion_coefficient_array[counter] = bessel_diffusion_coefficient
        counter += 1
    
    return amplitude_array,simulated_diffusion_array,Lifson_jackson_diffusion_coefficient_array,bessel_diffusion_coefficient_array

def run_simulation_noforce():
    A,S,LJ,B = compare()
    rot_array = np.ones(len(A))
    rot_array *= rotational_einstein_diff
    plt.plot(A,rot_array,label = 'Einstein diffusion')
    plt.plot(A,S,label = 'Simulated diffusion')
    plt.plot(A,LJ,label = 'Lifson_Jackson diffusion')
    plt.xlabel('Barrier amplitude [k_B T]')
    plt.ylabel('Diffusion coefficient [rad2/s] ')
    plt.legend()
    plt.show()
   
    
def run_theory():
    amplitude_array = np.linspace(0,10,1000)
    Lifson_jackson_diffusion_coefficient_array,bessel_diffusion_coefficient_array, = np.zeros(len(amplitude_array)),np.zeros(len(amplitude_array))
    counter = 0
    for amplitude in amplitude_array:
        _,Lifson_jackson_diffusion_coefficient,bessel_diffusion_coefficient = simulate(amplitude,nb_traj = 0,traj_length=1)
        Lifson_jackson_diffusion_coefficient_array[counter] = Lifson_jackson_diffusion_coefficient
        bessel_diffusion_coefficient_array[counter] = bessel_diffusion_coefficient
        counter += 1

    plt.plot(amplitude_array,Lifson_jackson_diffusion_coefficient_array,label = 'Lifson_Jackson diffusion')
    plt.plot(amplitude_array,bessel_diffusion_coefficient_array,label = 'Bessel approximation')
    plt.legend()
    plt.show()
    
run_simulation_noforce()


