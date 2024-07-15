import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from  Langevin_Class_v3 import *
from tqdm import tqdm
from numba import jit
#from  Dynamic_Langevin import *
viridis = plt.cm.viridis


def load():
    data = torch.load('trajectory.pt')
    return data

def traj_array():
    tracking_data = load()
    x = tracking_data['cx'].numpy()
    y = tracking_data['cy'].numpy()
    return x,y

def plot():
    x,y = traj_array()
    plt.scatter(x[600000:708397],y[600000:708397],s=1)
    plt.show()

def convert_to_angle(x,y):
    return np.arctan2(x,y)

def mean_square_displacement(array):
    traj = np.unwrap(array)
    msd = []
    centered_array = traj - np.mean(traj)
    for j in range(1,int(len(centered_array))):
        M = np.mean((centered_array[:-j] - centered_array[j:])**2)
        msd.append(M)
    return np.array(msd)

def slope_fit(t,msd):
    slope, _ = np.polyfit(t, msd, 1)
    diffusion = slope/2
    return diffusion

def super_diffusion(x, D, alpha):
    return 2*D * x**alpha

def fit_super_diffusion(x_data, y_data):
    params, _ = curve_fit(super_diffusion, x_data, y_data)
    D_opt, alpha_opt = params
    return D_opt, alpha_opt



def exp_traj():

    x,y = traj_array()
    x,y = x[600000:708397]*space_ratio,y[600000:708397]*space_ratio
    shift_x,shift_y = np.mean(x),np.mean(y)
    x,y = x - shift_x,y - shift_y
    theta_traj = convert_to_angle(x,y)
    t = np.arange(0,len(theta_traj),1)*dt_s
    return t,theta_traj,x,y

@jit(nopython=True)
def find_opposite_index(theta_traj, theta_i, epsilon=np.pi/180):
    """
    Find indices where the angle is approximately theta_i + pi or theta_i - pi
    """
    target_angles = [(theta_i + np.pi) % (2 * np.pi), (theta_i - np.pi) % (2 * np.pi)]
    indices = []
    for i in range(len(theta_traj)):
        theta = theta_traj[i]
        for target_angle in target_angles:
            if abs(theta - target_angle) < epsilon:
                indices.append(i)
                break
    return indices

@jit(nopython=True)
def radial_distance_re_inner(t, theta_traj, x, y, epsilon):
    """
    Inner function to compute distances with Numba
    """
    distances = []
    for i in range(len(theta_traj)):
        theta_i = theta_traj[i]
        opposite_indices = find_opposite_index(theta_traj, theta_i, epsilon)
        if len(opposite_indices) > 0:
            for opp_idx in opposite_indices:
                dist = np.sqrt((x[i] - x[opp_idx]) ** 2 + (y[i] - y[opp_idx]) ** 2)
                distances.append(dist)
    return np.array(distances)

def radial_distance_re(epsilon=np.pi/20):
    """ 
    Find the mean radial distance --> mean distance between two points with a pi angle difference
    """
    t, theta_traj, x, y = exp_traj()
    distances = radial_distance_re_inner(t, theta_traj, x, y, epsilon)
    return np.mean(distances)/2

#exp_parameter:
T_K = 273 + 22
KT = T_K*1.3806452e-23
FPS = 2000
dt_s = 1/FPS
space_ratio = 0.095e-6 #m/px
colors = viridis(np.linspace(0, 1, 10))
p = LangevinSimulator(dt=dt_s)
p.KT = KT
viscosity_NS_m2 = 0.001
R_bead = (1.3/2)*1e-6 #m

"""
Using gamma equation from:
Catch bond drives stator mechanosensitivity in the bacterial flagellar motor
"""

d_cell = 5*1e-9 #distance between bead and cell surface 
ratio = R_bead/ (d_cell + R_bead)
radial_dist = 1.0932736e-06 / 2 #to be changed 

gamma_bead_rot = 8 * np.pi * viscosity_NS_m2 * R_bead**3

gamma_rot_correction = gamma_bead_rot / np.abs(1 - (1/8) * ratio**3)

gamma_bead_trans = 6 * np.pi * viscosity_NS_m2 * R_bead * radial_dist**2 

gamma_trans_correction = gamma_bead_trans / np.abs(1 - (9/16)*ratio + (1/8)*ratio**3)

Faxen_correction = False

if Faxen_correction:
    gamma_tot = gamma_rot_correction + gamma_trans_correction
else: 
    gamma_tot = gamma_bead_rot + gamma_bead_trans
D = KT/gamma_tot
p.D = D




t,theta_traj,x,y = exp_traj()
traj = np.unwrap(theta_traj)

    
    

def plot_traj():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = 0
    t,theta_traj,x,y = exp_traj()
    #plt.scatter(x,y,s=0.01,color=colors[6])
    plt.scatter(np.arange(len(theta_traj)),np.unwrap(theta_traj))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_old_msd():
# diffusion coefficient
    msd = mean_square_displacement(theta_traj)
    #remove_end_aberation
    msd = msd[:-int(len(msd)/20)]
    t = np.arange(0,len(msd),1)*dt_s
    diffusion,alpha = fit_super_diffusion(t,msd)
    plt.scatter(t,msd,s=1, label = f'Diffusion = {diffusion:.2f}, MSD')
    plt.plot(t,super_diffusion(t, diffusion, alpha), label = f'Diffusion = {diffusion:.2f} [rad^2.s⁻1],alpha = {alpha:.2f}, MSD',color = 'red')
    print("Diffusion_coefficient = ",diffusion,'alpha = ',alpha)
    plt.set_xlabel('t[s]')
    plt.set_ylabel('MSD [rad^2]')
    plt.legend()

def exp_msd_removed_mean(plot=True):
    p = LangevinSimulator(dt=dt_s)
    t,traj,x,y = exp_traj()
    
    t = np.arange(len(traj))*p.dt
    traj = traj - t*traj[-1]/t[-1] # Removing the mean speed
    msd_nbpt = 500
    time_end = 1/4
    msd = p.msd(traj, time_end = time_end, msd_nbpt = msd_nbpt, print_time=False)
    max_lagtime = int(len(traj) * time_end)
    total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])*p.dt
    if plot:
        p.configure_plots()
        plt.loglog(total_lag_time,msd)
        plt.xlabel('lag_time [s]')
        plt.ylabel('MSD [rad²]')
        plt.show()

    plt.show() 


def exp_msd_simulation_compared(traj, plot=False):
    """
    TO DO :
    - Produce traj with removed mean with low amplitude and sufficient torque (V0 = 1kT, torque = 15kT ?)
    - compute msd over simulation and overlap the exp_msd with it
    - fit D
    """
    p = LangevinSimulator(dt=dt_s)
    p.gamma = gamma_tot
    p.D = D
    
    
    t = np.arange(len(traj))*p.dt
    msd_nbpt = 500
    time_end = 1/4
    msd = p.msd(traj, time_end = time_end, msd_nbpt = msd_nbpt, print_time=False)
    max_lagtime = int(len(traj) * time_end)
    total_lag_time = np.unique([int(lag) for lag in np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt))])*p.dt
    if plot:
        p.configure_plots()
        plt.loglog(total_lag_time,msd,label='MSD exp')
        plt.loglog(total_lag_time,2*D*total_lag_time,linestyle='--',label='2*D*t')
        plt.xlabel('lag_time [s]')
        plt.ylabel('MSD [rad²]')
        plt.legend()
        plt.show()

    plt.show() 
    return total_lag_time,msd
#plot_traj()
#exp_msd_removed_mean_simulation_compared(plot=True)


def find_constant_derivative_slices(trajectory, min_length, plot=False, tolerance=1):
    # Calculate the derivative of the trajectory
    derivative = np.diff(trajectory)
    
    # Initialize the list of slices
    slices = []
    start_index = 0

    # Iterate through the derivative array
    for i in range(1, len(derivative)):
        # Check if the change in derivative is within the tolerance
        if abs(derivative[i] - derivative[i-min_length]) > tolerance:
            # If the change exceeds the tolerance, create a slice
            slices.append((start_index, i))
            start_index = i
    
    # Add the last slice
    slices.append((start_index, len(trajectory) - 1))
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory, label='Trajectory')
        
        # Add vertical lines for slice boundaries
        for start, end in slices:
            plt.axvline(x=start, color='r', linestyle='--')
            #plt.axvline(x=end, color='r', linestyle='--')
        
        plt.xlabel('t [s]')
        plt.ylabel('Trajectory angle unwrapped')
        plt.title('Trajectory with Slice Boundaries')
        plt.legend()
        plt.show()
    
    return slices 

#slices_index = find_constant_derivative_slices(traj,100,plot=True,tolerance=0.076)

def constant_len_slices(trajectory, slice_length, plot=False):
    # Initialize the list of slices
    slices = []
    start_index = 0

    # Iterate through the trajectory array and create slices of constant length
    for i in range(slice_length, len(trajectory), slice_length):
        slices.append((start_index, i))
        start_index = i

    # Add the last slice if it's not empty
    if start_index < len(trajectory):
        slices.append((start_index, len(trajectory) - 1))

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory, label='Trajectory')

        # Add vertical lines for slice boundaries
        for start, end in slices:
            plt.axvline(x=start, color='r', linestyle='--')
            plt.axvline(x=end, color='r', linestyle='--')

        plt.xlabel('Index')
        plt.ylabel('Trajectory Value')
        plt.title('Trajectory with Constant Length Slice Boundaries')
        plt.legend()
        plt.show()

    return slices

frequency=26


def plot_traj_slices(traj, slice_length, plot=True, plot_mean_msd = True,plot_msd_diff = True):
    slices_index = constant_len_slices(traj, slice_length, plot=True)
    
    traj_slice_box = []
    
    
    for (i, j) in slices_index:
        print(i,j)
        
        traj_slice = traj[i:j]  
        traj_slice -= traj_slice[1]
        t = np.arange(len(traj_slice)) * dt_s
        print('slope', traj_slice[-1] / t[-1] )
        traj_slice = traj_slice - t * traj_slice[-1] / t[-1] 
         
        traj_slice_box.append(traj_slice)

    # Merge the slices
    merged_trajectory = np.concatenate(traj_slice_box)
    total_time = np.arange(len(merged_trajectory)) * dt_s
    
    if plot:
    # Plot the merged trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(total_time, merged_trajectory, label='Merged Trajectory')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [╬rad')
        plt.title('Merged Trajectory with Linear Trend Removed')
        plt.legend()
        plt.show()
        
    
    traj_slice_box.pop()
    msd_box = []
    [msd_box.append(exp_msd_simulation_compared(traj = traj_slice_box[i],plot=False)) for i in range(len(traj_slice_box))]
    mean_msd = np.mean(msd_box[1],axis=0)
    total_lag_time = msd_box[0][0]
        
    if plot_mean_msd:    
        plt.loglog(total_lag_time,mean_msd,label=f'MSD, slice_duration = {slice_length*dt_s:.1}')
        plt.loglog(total_lag_time,2*D*total_lag_time,linestyle='--',label='2*D*t')
        plt.xlabel('lag_time [s]')
        plt.ylabel('MSD [rad²]')
        plt.legend()
        
    if plot_msd_diff:
        diff_msd = np.diff(mean_msd,1)
        plt.loglog(total_lag_time[:-1],diff_msd,label='MSD_diff')
        plt.xlabel('lag_time [s]')
        plt.ylabel('diff_MSD [rad²/s]')
        plt.legend()
    return traj_slice_box

plot_traj_slices(traj,10000)








def analytical_potential(x, V0 = float, torque = float):
    U =  V0 / 2 * np.sin(x * frequency) - torque * x / (2 * np.pi)
    print('UUUUUUUUUUUUUUUUUU',type(U),U)
    return U

def ReimanD_eff(V0, F):
    
    def U(x):
        return V0 / 2 * np.sin(x * frequency) - F * x / (2 * np.pi)

    L = 2 * np.pi / frequency  # Period of the potential
    D0 = D  # Free diffusion coefficient

    def I_plus(x):
        return quad(lambda y: np.exp(U(x) - U(y) - F * (x - y)), 0, L)[0] / D0

    def I_minus(x):
        return quad(lambda y: np.exp(-(U(x) - U(y) - F * (x - y))), 0, L)[0] / D0

    x_values = np.linspace(0, L, 100)
    I_plus_values = [I_plus(x) for x in x_values]
    I_minus_values = [I_minus(x) for x in x_values]

    I_plus_integral = np.trapz(I_plus_values, x_values) / L
    I_minus_integral = np.trapz(I_minus_values, x_values) / L

    D_eff = D0 * I_minus_integral / (I_plus_integral ** 3)
    return D_eff



"""
def plot_theoretical_Deff_range_V0():
    V0_box = np.arange(0,5,0.2)
    
    frequency = 26
    F_c_max = 5 *frequency*np.pi
    for V0 in V0_box:
        F_c = frequency * V0 * np.pi
        F_box = np.linspace(0, 1.5 * F_c_max, 50)
        D_effs = []
        for torque in F_box:
            p.D = D
            p = LangevinSimulator(dt=1e-5,torque=torque,frequency=frequency)
            D_effs.append(p.D_eff_reimann(V0)/D)
        plt.plot(F_box, D_effs, label=f'D_eff, V0={V0:.1f}kT')
    
    plt.xlabel('Torque')
    plt.ylabel('D_eff')
    #plt.legend()
    plt.show()
"""
      




#plot_traj_constant_der()









