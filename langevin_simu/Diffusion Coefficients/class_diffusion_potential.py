import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
        self.torque = torque*self.T_K*self.k_b

    def generate_seq(self, N):
        standard_deviation = 1 
        W_seq = np.random.normal(0, standard_deviation, N)
        return W_seq

    def mean_square_displacement(self, traj, time_end, time_skip):
        lag_time = np.arange(0, int(len(traj) * time_end), time_skip)
        msd = []
        for j in lag_time:
            if j == 0 :
                msd.append(0)
                continue
            msd.append(np.mean((traj[:-j] - traj[j:])**2))
        return np.array(msd),lag_time
    

    def periodic_potential(self, A, x):
        return A * np.cos(x * self.frequency)

    def static_process(self, N, A):
        x = 0
        stored_position = np.zeros(N)
        w = self.generate_seq(N)
        for i in np.arange(0, N):
            dx = -(1 / self.rotational_drag) * self.dt_s * (self.periodic_potential(A, x + self.space_step) - self.periodic_potential(A, x)) / self.space_step
            dx = dx + self.torque + np.sqrt(2 * self.rotational_einstein_diff * self.dt_s) * w[i]
            x = np.mod(x + dx, 2 * np.pi)
            stored_position[i] = x
            dx = 0
        return stored_position

    def msd_W_trajs(self, W, N, amplitude):
        msd_list = np.arange(1,W)
        mean_msd = np.zeros(N - 1)
        for j in range(W):
            traj = np.unwrap(self.static_process(N, amplitude * self.k_b * self.T_K))
            msd_list[j] = self.mean_square_displacement(traj)
        for msd in msd_list:
            mean_msd = np.add(mean_msd, msd)
        mean_msd *= 1 / W
        mean_msd = mean_msd[:-int(len(mean_msd) / 2)]
        return mean_msd
    
    def msd_in_matrix(self, W, N, amplitude,time_end,time_skip):
        msd_matrix = []
        for j in range(W):
            traj = np.unwrap(self.static_process(N, amplitude * self.k_b * self.T_K))
            interm,_ = self.mean_square_displacement(traj,time_end,time_skip)
            msd_matrix.append(interm)
        return msd_matrix
        
        
    def mean_msd_with_theoretical_variance(self, W, N, amplitude):
        mean_msd = self.msd_W_trajs(W, N, amplitude)
        var_array = []
        for n in range (1,len(mean_msd)): # n is the lag time
            var = ((2*self.rotational_einstein_diff*n*self.dt_s)**2)*(2*n*n + 1)/(3*n*(N-n+1))
            var_array.append(var) # ?
        error_sizes = np.sqrt(var_array)/50
        return mean_msd, error_sizes
        

    def error_bar(self, x_values, y_values, error_sizes):
        plt.errorbar(x_values, y_values, yerr=error_sizes, fmt='o', capsize=5, label='Data with Error Bars')        
    
    def slope_fit(self, t, mean_msd):
        slope, _ = np.polyfit(t, mean_msd, 1)
        simulated_diffusion = slope / 2
        return simulated_diffusion
    
    def super_diffusion(self,t, D):
        return D*t + (self.torque/self.rotational_drag)*t*t

    
    def fit_super_diffusion(self,x_data, y_data):
        params, _ = curve_fit(self.super_diffusion, x_data, y_data)
        D_opt = params
        return D_opt
    
    """
    Lifson and Jackson functions
    Should not be dependent from frequency
    """

    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / self.frequency, np.pi / self.frequency, args=(amplitude))
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

    def LJ_mean_denominator(self, amplitude_array = np.linspace(0, 5, 1000)):
        LJ_array = np.zeros(len(amplitude_array))
        counter1 = 0
        theta = np.linspace(0, 2*np.pi, 1000)
        for amplitude in amplitude_array:
            counter0 = 0
            denominator1 = np.zeros(len(theta))
            denominator2 = np.zeros(len(theta))
            for angle in theta:
                denominator1[counter0] = self.integrand1(angle,amplitude)
                denominator2[counter0] = self.integrand1(angle,-amplitude)
                counter0 += 1
            LJ_array[counter1] = self.rotational_einstein_diff/ (np.mean(denominator1) * np.mean(denominator2))
            #*(np.sinh(self.torque*(2*np.pi / self.frequency)/2))/(self.torque*(2*np.pi / self.frequency)/2) 
            counter1 += 1
        return amplitude_array,LJ_array


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


   