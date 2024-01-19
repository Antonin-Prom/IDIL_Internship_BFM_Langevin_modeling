import numpy as np




class Brownian :
    def __init__(self,dt,energy_argument,custom_energy):

        #finite spatial step
        self.d = 1e-12
        # timesteps
        self.dt = dt
        #physical parameters
        self.kT = 1.38e-23*300
        self.m_kg = 1.1e-14 
        R = 1e-6
        eta = 1e-3
        self.load = 6*np.pi*eta*R
        self.diffusion = self.kT/self.load
        self.sd = np.sqrt(2*self.diffusion*self.dt)
        self.tau = self.m_kg/self.load
        self.stored_position = []
        self.stored_time = []
        self.stored_potential = []
        self.custom_energy = custom_energy
        self.energy_argument = energy_argument
        self.x = 0

    def static_energy(self,x):
        return self.custom_energy(self.energy_argument,x)
    
    def static_process(self,N):
        time = 0
        for i in np.arange(0,N):
            dx = -(1/self.load)*self.dt*(self.static_energy(self.x+self.d) - self.static_energy(self.x))/self.d
            dx = dx + np.random.normal(0.0,self.sd)
            self.x = self.x + dx
            time = time + self.dt
            self.stored_position.append(self.x)
            dx = 0
        return self.stored_position
    
    def dynamic_energy(self,x,t):
        return np.sin(t)*self.custom_energy(self.energy_argument,x)

    def dynamic_process(self):
        time = 0
        for i in np.arange(0,self.timesteps):
            dx = -(1/self.load)*self.dt*(self.dynamic_energy(self.x+self.d,time) - self.dynamic_energy(self.x,time))/self.d
            dx = dx + np.random.normal(0.0,self.sd)
            self.x = self.x + dx
            time = time + self.dt
            potential = self.dynamic_energy(self.x,time)
            self.stored_position.append(self.x)
    
            dx = 0
        return self.stored_position
        
    def generate_randseq(self,N):
        standard_deviation = 1
        # Array of random numbers following a Gaussian distribution with 0 mean and standard deviation of 1  
        W_seq = np.array(np.random.normal(0,standard_deviation,N))
        return W_seq

    def walk_diffusion(self,N,w):
        x = np.zeros(N)
        for i in range(1,N):
            x[i] = x[i-1] + np.sqrt(2*self.diffusion*self.dt)*w[i]
        return x

    def walk_inertia(self,N,w):
        x = np.zeros(N)
        # As a term in [i-2] is needed to compute x[i] we must define x[1] first.
        x[1] = ((2+self.dt*(1/self.tau))/(1+self.dt*(1/self.tau)))*x[0] + (np.sqrt(2*self.kT*self.load)/(self.m_kg*(1 + self.dt*(1/self.tau))))*self.dt**(3/2)*w[1]
        for i in range(2,N):
            x[i] = ((2+self.dt*(1/self.tau))/(1+self.dt*(1/self.tau)))*x[i-1] - (1/(1+self.dt*(1/self.tau)))*x[i-2] + (np.sqrt(2*self.kT*self.load)/(self.m_kg*(1 + self.dt*(1/self.tau))))*self.dt**(3/2)*w[i]
        return x
    
    def walk_potential_well(self,N,kx):
        w_x = self.generate_randseq(N)
        x = np.zeros(N)

        for i in range(1,N):
            x[i] = x[i-1] - (1/self.load)*kx*x[i-1]*self.dt + np.sqrt(2*self.diffusion*self.dt)*w_x[i]
        return x
    
    def velocity_auto_correlation(self,array):

        position = array
        velocities = np.diff(position, axis=0)
        centered_velocities = velocities - np.mean(velocities)
        autocorr = np.convolve(centered_velocities, centered_velocities[::-1], mode='full')
        autocorr /= autocorr[len(autocorr)//2]  
        return autocorr

    def mean_square_displacement(self,array):

        msd = []
        centered_array = array - np.mean(array)

        for j in range(1,int(len(centered_array))):
            M = np.mean((centered_array[:-j] - centered_array[j:])**2)
            msd.append(M)

        return np.array(msd)


