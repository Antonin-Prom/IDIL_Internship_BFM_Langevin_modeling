import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class DiffusionSimulation:
    def __init__(self,frequency = 26, torque = 0, dt=1e-12):
        # Constants
        self.T_K = 300
        self.k_b = 1.3806452e-23
        self.R_m = 1e-6
        self.m_kg = 1.1e-14 
        self.viscosity_NS_m2 = 0.001
        self.load = 6 * np.pi * self.viscosity_NS_m2 * self.R_m 
        self.rotational_drag = 8 * np.pi * self.viscosity_NS_m2 * self.R_m**3
        self.moment_inertia = (2/5)*self.m_kg*self.R_m**2
        self.tau = self.moment_inertia / self.rotational_drag
        self.einstein_diffusion = self.k_b * self.T_K / self.load
        self.rotational_einstein_diff = self.k_b * self.T_K / self.rotational_drag
        self.dt_s = dt # tau
        self.frequency = frequency
        self.space_step = 1e-12
        self.torque = torque*self.T_K*self.k_b

    def generate_seq(self, N):
        standard_deviation = 1 
        W_seq = np.random.normal(0, standard_deviation, N)
        return W_seq

    def mean_square_displacement(self, traj, time_end, time_skip):
        """Compute the mean square displacement by iterating through an array of lag times

        Args:
            traj (array): The targeted array of positions.
            time_end (float): The iterated fraction of the trajectory length.
            time_skip (int): Quantity of points skiped when performing msd.

        Returns:
            array: The mean square displacement
            array: The lag times array    
        """
        lag_time = np.arange(0, int(len(traj) * time_end), time_skip)
        msd = []
        for j in lag_time:
            if j == 0 :
                msd.append(0)
                continue
            msd.append(np.mean((traj[:-j] - traj[j:])**2))
        return np.array(msd),lag_time
    

    def tilted_periodic_potential(self, A, x):
        return self.torque*x + A * np.cos(x * self.frequency)

    def static_process(self, N, A):
        """ Perform the overdamped rotational langevin dynamic simulation in a given potential, all units are in S.I.

        Args:
            N (int): trajectory length
            A (float): barrier amplitude 

        Returns:
            array: angular trajectory
        """
        A *= self.k_b * self.T_K
        x = 0
        stored_position = np.zeros(N)
        w = self.generate_seq(N)
        for i in np.arange(0, N):
            dx = -(1 / self.rotational_drag) * self.dt_s * (self.tilted_periodic_potential(A, x + self.space_step) - self.tilted_periodic_potential(A, x)) / self.space_step + np.sqrt(2 * self.rotational_einstein_diff * self.dt_s) * w[i]
            x = np.mod(x + dx, 2 * np.pi)
            stored_position[i] = x
        return stored_position


    def proceed_traj1(self, N, A):
        """ Perform the overdamped rotational Langevin dynamic simulation in a given potential, all units are in S.I.

        Args:
            N (int): trajectory length
            A (float): barrier amplitude 

        Returns:
            array: angular trajectory
        """
        w = self.generate_seq(N)
        A *= self.k_b * self.T_K
        x = 0 
        positions = [ x := np.mod((x - (1 / self.rotational_drag) * self.dt_s * (self.tilted_periodic_potential(A, x + self.space_step) - self.tilted_periodic_potential(A, x)) / self.space_step + np.sqrt(2 * self.rotational_einstein_diff * self.dt_s) * w[i]),2*np.pi) for i in range(N)]
        return positions    
    
    def msd_in_matrix(self, W, N, amplitude,time_end,time_skip):
        """ Compute multiple trajectories than perform mean square displacement and return it in a matrix

        Args:
            W (float): Number of msd to compute.
            N (_type_): Number of trajectory points.
            amplitude (float): Barrier amplitude.
            time_end (float): The iterated fraction of the trajectory length.
            time_skip (int): Quantity of points skiped when performing msd.

        Returns:
            list: matrix of msd of size (W,N-1)
        """
        msd_matrix = []
        for j in range(W):
            traj = np.unwrap(self.static_process(N, amplitude ))
            interm,_ = self.mean_square_displacement(traj,time_end,time_skip)
            msd_matrix.append(interm)
        return msd_matrix
        
        
    def mean_msd_with_theoretical_variance(self, W, N, amplitude,time_end,time_skip):
        """Compute the theoretical standard deviation of the mean square displacement with the formula (13) from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1260142/pdf/biophysj00109-0164.pdf
        
        Args:
            W (float): Number of msd to compute.
            N (_type_): Number of trajectory points.
            amplitude (float): Barrier amplitude.
            time_end (float): The iterated fraction of the trajectory length.
            time_skip (int): Quantity of points skiped when performing msd.

        Returns:
            array: array of meaned msd
            array: array of theoretical standard deviation
        """
        mean_msd = np.mean(self.msd_in_matrix(W, N, amplitude,time_end,time_skip), axis = 0)
        var_array = []
        for n in range (1,len(mean_msd)): # n is the lag time
            var = ((2*self.rotational_einstein_diff*n*self.dt_s)**2)*(2*n*n + 1)/(3*n*(N-n+1))
            var_array.append(var) # ?
        error_sizes = np.sqrt(var_array)/50
        return mean_msd, error_sizes
        

    def slope_fit(self, t, mean_msd):
        slope, _ = np.polyfit(t, mean_msd, 1)
        simulated_diffusion = slope / 2
        return simulated_diffusion
    
    def parabola_fit(self, t, mean_msd):
        a, b, c = np.polyfit(t, mean_msd, 2)
        return a,b,c
    
    def super_diffusion(self,t, D):
        return D*t + (self.torque/self.rotational_drag)*t*t
    
    def fit_super_diffusion(self,x_data, y_data):
        D_opt, covar = curve_fit(self.super_diffusion, x_data, y_data)
        std = np.sqrt(np.diag(covar))
        return D_opt,std
    
    """
    Lifson and Jackson methods
    """

    def factor1(self, amplitude):  
        result, _ = quad(self.integrand1, -np.pi / self.frequency, np.pi / self.frequency, args=(amplitude))
        return result

    def integrand1(self, x, amplitude):
        return np.exp(self.tilted_periodic_potential(amplitude, x))

    def lifson_jackson_noforce(self, amplitude):
        lifson_jackson1 = self.rotational_einstein_diff * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))
        return lifson_jackson1 
    
    def lifson_jackson_force(self, amplitude):
        lifson_jackson2 = (self.rotational_einstein_diff * (2 * np.pi / self.frequency)**2 / ((self.factor1(amplitude)) * (self.factor1(-amplitude)))) * (np.sinh(self.torque*(2*np.pi / self.frequency)/2))/(self.torque*(2*np.pi / self.frequency)/2)
        return lifson_jackson2
    
    def Lifson_Jack_vs_amp_noforce(self, amplitude_array = np.linspace(0, 5, 1000)):
        lifson_jackson_diffusion_coefficient_array = np.zeros(len(amplitude_array))
        counter = 0
        for amplitude in amplitude_array:
            lifson_jackson_diffusion_coefficient_array[counter] = self.lifson_jackson_noforce(amplitude)
            counter += 1
        return amplitude_array, lifson_jackson_diffusion_coefficient_array

    def LJ_mean_denominator(self, amplitude_array = np.linspace(0, 5, 1000)):
        """ Compute the effective Lifson & Jackson diffusion coefficient, average is perform by np.mean and not integration.
        

        Args:
            amplitude_array (array, optional): array of amplitude on wich diffusion coefficient will be computed. Defaults to np.linspace(0, 5, 1000).

        Returns:
            array: amplitude array
            array: diffusion coefficient array                                                                                                                                                                                                                                                                                                                      
        """
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

   