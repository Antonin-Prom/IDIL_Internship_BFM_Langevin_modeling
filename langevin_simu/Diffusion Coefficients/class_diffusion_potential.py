import numpy as np
from scipy.integrate import quad

class DiffusionSimulation:
    def __init__(self,frequency = 26, torque = 0):
        # Constants
        self.T_K = 300
        self.k_b = 1.3806452e-23
        self.R_m = 1e-6
        self.m_kg = 1.1e-14 
        self.viscosity_NS_m2 = 0.001
        self.load = 6 * np.pi * self.viscosity_NS_m2 * self.R_m 
        self.rotational_drag = 8 * np.pi * self.viscosity_NS_m2 * self.R_m**3
        self.tau = self.m_kg / self.load
        self.einstein_diffusion = self.k_b * self.T_K / self.load
        self.rotational_einstein_diff = self.k_b * self.T_K / self.rotational_drag
        self.dt_s = 100 * self.tau
        self.frequency = frequency
        self.space_step = 1e-12
        self.torque = torque

    def generate_seq(self, N):
        standard_deviation = 1 
        W_seq = np.random.normal(0, standard_deviation, N)
        return W_seq

    def mean_square_displacement(self, array):
        msd = []
        centered_array = array - np.mean(array)
        for j in range(1, int(len(centered_array))):
            M = np.mean((centered_array[:-j] - centered_array[j:])**2)
            msd.append(M)
        return np.array(msd)

    def periodic_potential(self, A, x):
        return A * np.cos(x * self.frequency)

    def static_process(self, N, A):
        x = 0
        stored_position = []
        w = self.generate_seq(N)
        for i in np.arange(0, N):
            dx = -(1 / self.rotational_drag) * self.dt_s * (self.periodic_potential(A, x + self.space_step) - self.periodic_potential(A, x)) / self.space_step
            dx = dx + self.torque + np.sqrt(2 * self.rotational_einstein_diff * self.dt_s) * w[i]
            x = np.mod(x + dx, 2 * np.pi)
            stored_position.append(x)
            dx = 0
        return stored_position

    def msd_W_trajs(self, W, N, amplitude):
        msd_list = []
        mean_msd = np.zeros(N - 1)
        for j in range(W):
            traj = np.unwrap(self.static_process(N, amplitude * self.k_b * self.T_K))
            msd_list.append(self.mean_square_displacement(traj))
        for msd in msd_list:
            mean_msd = np.add(mean_msd, msd)
        mean_msd *= 1 / W
        mean_msd = mean_msd[1000:-int(len(mean_msd) / 2)]
        return mean_msd

    def slope_fit(self, t, mean_msd):
        slope, _ = np.polyfit(t, mean_msd, 1)
        simulated_diffusion = slope / 2
        return simulated_diffusion

    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / 26, np.pi / 26, args=(amplitude))
        return result

    def integrand1(self, x, amplitude):
        return np.exp(self.periodic_potential(amplitude, x))

    def lifson_jackson_noforce(self, amplitude):
        lifson_jackson1 = self.rotational_einstein_diff * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))
        return lifson_jackson1 
    
    def lifson_jackson_force(self, amplitude):
        lifson_jackson2 = self.rotational_einstein_diff * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude))) * (np.sinh(self.torque*(2*np.pi / self.frequency)/2))/(self.torque*(2*np.pi / self.frequency)/2)
        return lifson_jackson2

    def Bessel(self, A):
        I0 = np.exp(A) / (np.sqrt(2 * np.pi * A)) 
        D = self.rotational_einstein_diff / (I0 * I0)
        return D

    def simulate(self, amplitude, nb_traj, traj_length):
        mean_msd = self.msd_W_trajs(nb_traj, traj_length, amplitude)
        t = np.arange(len(mean_msd)) * self.dt_s
        simulated_diffusion = self.slope_fit(t, mean_msd)
        return simulated_diffusion
    
    def Lifson_Jack_vs_amp_noforce(self, amplitude_array = np.linspace(0, 5, 1000)):
        lifson_jackson_diffusion_coefficient_array = np.zeros(len(amplitude_array))
        counter = 0
        for amplitude in amplitude_array:
            lifson_jackson_diffusion_coefficient_array[counter] = self.lifson_jackson_noforce(amplitude)
            counter += 1
        return amplitude_array, lifson_jackson_diffusion_coefficient_array


    def compare(self,amplitude_array = np.linspace(0, 5, 50),nb_traj= 2, traj_length=5000):
        simulated_diffusion_array, lifson_jackson_diffusion_coefficient_array, bessel_diffusion_coefficient_array = np.zeros(len(amplitude_array)), np.zeros(len(amplitude_array)), np.zeros(len(amplitude_array))
        counter = 0
        for amplitude in amplitude_array:
            simulated_diffusion, lifson_jackson_diffusion_coefficient, bessel_diffusion_coefficient = self.simulate(amplitude, nb_traj, traj_length)
            simulated_diffusion_array[counter] = simulated_diffusion
            lifson_jackson_diffusion_coefficient_array[counter] = lifson_jackson_diffusion_coefficient
            bessel_diffusion_coefficient_array[counter] = bessel_diffusion_coefficient
            counter += 1

        return amplitude_array, simulated_diffusion_array, lifson_jackson_diffusion_coefficient_array, bessel_diffusion_coefficient_array


   