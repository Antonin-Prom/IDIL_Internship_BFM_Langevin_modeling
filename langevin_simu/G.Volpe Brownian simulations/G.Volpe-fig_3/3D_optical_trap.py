import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Physical parameters for a micro bead in water at R.T

T_K = 300
k_b = 1.3806452e-23
R_m = 1e-6
m_kg = 1.1e-14 
viscosity_NS_m2 = 0.001
load = 6*np.pi*viscosity_NS_m2*R_m 
tau = m_kg/load
diffusion = k_b*T_K/load
kx = 1*(1)*(10e-15)/(10e-9) #10e-6 femto_Newton/nm
ky = 1*(1)*(10e-15)/(10e-9)
kz = 0.2*(1)*(10e-15)/(10e-9)
ratio_phi = kx/load
print('ratio_phi',ratio_phi)
print('tau',tau)
# Simulation parameters
dt_s = 10000*tau
print('dt',dt_s)



class traj_3D :

    def __init__(self,N,nb_traj):
        self.N = N
        self.nb_traj = nb_traj
    
    def generate_randseq(self):
        standard_deviation = 1
        # Array of random numbers following a Gaussian distribution with 0 mean and standard deviation of 1  
        W_seq = np.array(np.random.normal(0,standard_deviation,self.N))
        return W_seq

    def walk(self):
        N = self.N
        w_x = self.generate_randseq()
        w_y = self.generate_randseq()
        w_z = self.generate_randseq()
        x = np.zeros(N)
        y = np.zeros(N)
        z = np.zeros(N)
        for i in range(1,N):
            x[i] = x[i-1] - (1/load)*kx*x[i-1]*dt_s + np.sqrt(2*diffusion*dt_s)*w_x[i]
            y[i] = y[i-1] - (1/load)*ky*y[i-1]*dt_s + np.sqrt(2*diffusion*dt_s)*w_y[i]    
            z[i] = z[i-1] - (1/load)*kz*z[i-1]*dt_s + np.sqrt(2*diffusion*dt_s)*w_z[i]
               
        return x,y,z

    def plot3Dtraj(self):

        N = self.N
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x,y,z = self.walk()

        ax.plot(x, y, z)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)    
        plt.show()

    def distribution_2D(self):
        nb_traj = self.nb_traj
        N = self.N
        x_cloud = []
        y_cloud = []
        z_cloud = []
        for j in range(nb_traj):
            x,y,z = self.walk()
            #x_cloud, y_cloud, z_cloud = zip(*[(i, j, k) for i, j, k in zip(x, y, z)])
            for i in range(N):
                x_cloud.append(x[i])
                y_cloud.append(y[i])
                z_cloud.append(z[i])

        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.scatter(x_cloud,y_cloud,s=1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title("xy")
        ax2.scatter(x_cloud,z_cloud,s=1)
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')      
        ax2.set_title("xz")  
        ax3.scatter(y_cloud,z_cloud,s=1)
        ax3.set_xlabel('y')
        ax3.set_ylabel('z')
        ax3.set_title("yz")
        
        plt.show()

duration = 5
N = int(duration/dt_s)
print(N)
traj1 = traj_3D(N,2000)
traj1.distribution_2D()