import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load():
    data = torch.load('/home/antonin/Documents/GitHub/IDIL_Internship_BFM_Langevin_modeling/exp_data/trajectory.pt')
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



#exp_parameter:
T_K = 273 + 22
FPS = 2000
dt_s = 1/FPS
space_ratio = 0.095e-6 #m/px

# initialising array
x,y = traj_array()
x,y = x[600000:708397]*space_ratio,y[600000:708397]*space_ratio
shift_x,shift_y = np.mean(x),np.mean(y)
x,y = x - shift_x,y - shift_y

theta_traj = convert_to_angle(x,y)
t = np.arange(0,len(theta_traj),1)*dt_s

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(x,y,s=1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

# diffusion coefficient
msd = mean_square_displacement(theta_traj)
#remove_end_aberation
msd = msd[:-int(len(msd)/20)]
t = np.arange(0,len(msd),1)*dt_s
diffusion,alpha = fit_super_diffusion(t,msd)
ax2.scatter(t,msd,s=1, label = f'Diffusion = {diffusion:.2f}, MSD')
ax2.plot(t,super_diffusion(t, diffusion, alpha), label = f'Diffusion = {diffusion:.2f} [rad^2.s⁻1],alpha = {alpha:.2f}, MSD',color = 'red')
print("Diffusion_coefficient = ",diffusion,'alpha = ',alpha)
ax2.set_xlabel('t[s]')
ax2.set_ylabel('MSD [rad^2]')
ax2.legend()

plt.show() 



