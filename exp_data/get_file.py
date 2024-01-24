import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

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


#exp_parameter:
T_K = 273 + 22
FPS = 10000
dt_s = 1/FPS

# initialising array
x,y = traj_array()
x,y = x[600000:708397],y[600000:708397]
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
t = np.arange(0,len(msd),1)*dt_s
diffusion = slope_fit(t,msd)
ax2.scatter(t,msd,s=1, label = f'Diffusion = {diffusion:.2f}, MSD')
print(diffusion)
ax2.set_xlabel('t')
ax2.set_ylabel('MSD [pixel^2]')
ax2.legend()

plt.show() 



