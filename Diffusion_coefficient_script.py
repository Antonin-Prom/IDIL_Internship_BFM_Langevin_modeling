import numpy as np
import matplotlib.pyplot as plt




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

def factor1(x,amplitude):
    return np.exp(periodic_potential(amplitude,x))

def factor2(x,amplitude):
    h = 1e-12
    return (np.exp(- periodic_potential(amplitude,x+h))-np.exp(-periodic_potential(amplitude,x-h)))/(2*h)

def lifson_jackson(amplitude):
    X = np.linspace(0,1/frequency,10000)
    lifson_jackson1 = rotational_einstein_diff/(np.mean(factor1(X,amplitude))*np.mean(factor2(X,amplitude)))
    return lifson_jackson1 

def Bessel(A):
    I0 = np.exp(A)/(np.sqrt(2*np.pi*A)) 
    D = rotational_einstein_diff/(I0*I0)
    return D

def simulate(amplitude,nb_traj,traj_length):
    mean_msd = msd_W_trajs(nb_traj,traj_length,amplitude)
    t = np.arange(len(mean_msd))*dt_s
    simulated_diffusion = slope_fit(t,mean_msd)
    Lifson_jackson_diffusion_coefficient = 0#lifson_jackson(amplitude)
    bessel_diffusion_coefficient = 0#Bessel(amplitude)
    return simulated_diffusion,Lifson_jackson_diffusion_coefficient,bessel_diffusion_coefficient

def compare():
    amplitude_array = np.linspace(0,1e-5,100)
    simulated_diffusion_array,Lifson_jackson_diffusion_coefficient_array,bessel_diffusion_coefficient_array, = np.zeros(len(amplitude_array)),np.zeros(len(amplitude_array)),np.zeros(len(amplitude_array))
    counter = 0
    for amplitude in amplitude_array:
        simulated_diffusion,Lifson_jackson_diffusion_coefficient,bessel_diffusion_coefficient = simulate(amplitude,nb_traj = 50,traj_length=5000)
        simulated_diffusion_array[counter] = simulated_diffusion
        Lifson_jackson_diffusion_coefficient_array[counter] = Lifson_jackson_diffusion_coefficient
        bessel_diffusion_coefficient_array[counter] = bessel_diffusion_coefficient
        counter += 1
    
    return amplitude_array,simulated_diffusion_array,Lifson_jackson_diffusion_coefficient_array,bessel_diffusion_coefficient_array

def run():
    
    A,S,LJ,B = compare()
    
    rot_array = np.ones(len(A))
    rot_array *= rotational_einstein_diff
    simu_mean = np.mean(S)
    plt.hist(S,density = True)
    plt.axvline(x=rotational_einstein_diff, color='green', label='Einstein diffusion')
    plt.axvline(x=simu_mean, color='red', linestyle='--', label='Mean of simulated diffusion')
    plt.xlabel('Diffusion coefficient [rad2/s]')
    plt.ylabel('Density ')
    plt.legend()
    plt.show()
    print(S)
run()




