''' simulations of 1D angular tilted period potential '''

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time





@njit
def _make_trace(x, npts, dUdx, dt, gamma, thermnoise):
    ''' part of the Langevin simulation, called by TiltedPeriodicDiffusion.make_trace() for numba acceleration '''
    for i in range(npts - 1):
        # idx of dUdx(x[i]):
        dUdx_idxi = int(np.round(x[i]*len(dUdx)/(2*np.pi)) - 1)
        force = -dUdx[dUdx_idxi]
        # Langevin eq:
        x[i+1] = x[i] + force*dt/gamma + thermnoise[i]
    return x





class TiltedPeriodicDiffusion():
    ''' 
    ex: 
        # generate one trace and its plots:
        pp = tilted_periodic_diffusion.TiltedPeriodicDiffusion(dt=1e-4, gamma=2.5e-20)
        U = pp.make_potential_sin(pp.x_pot, ampl=1, period=10, tilt=0, plots=0)
        pp.make_trace(U, npts=1e6, plots=1)

        # repeated MSD and fit:
        pp = tilted_periodic_diffusion.TiltedPeriodicDiffusion()
        msd_t, msd_mean, msd_sdt, msd_stack = pp.make_sim_avg_MSD(repetitions=10, trace_npts=100000, pot_ampl=0, pot_tilt=0, plots=1)
        pp.fit_msd(msd_t, msd_mean, plots=1)
    ''' 
    Kb = 1.38e-23                                       # Boltzmann const. [Nm/K]
    T = 300                                             # temperature
    KT = Kb*T
    eta = 0.001                                         # water viscosity [Pa*s] = [N*s/m^2] 
    x_pot = np.linspace(0, 2*np.pi, 50000)              # periodic potential spatial variable 



    def __init__(self, dt=1e-6, bead_radius=100e-9, gamma=None):
        self.dt = dt                                                            # simulation time step [s]
        self.bead_radius = bead_radius                                          # [m]
        gamma_lin = 6*np.pi*self.eta*self.bead_radius                                     # bead translational drag [Ns/m]
        gamma_rot_parallel = 8*np.pi*self.eta*self.bead_radius**3                         # [Nms] bead rotational drag, axis perpendicular (yes) to surface
        self.L = 100e-9                                                         # [m] cylinder length
        if not gamma:
            self.gamma = 3.841*np.pi*self.eta*self.L*self.bead_radius**2*(1+0.3)    # [Nsm] cylinder gamma_rot_parallel for diam=0.5*length (see SI our PhFM paper)
        else:
            self.gamma = gamma
        self.D = self.KT/self.gamma                                             # diff. coeff. m^2/s



    def run(self):
        ''' just a convenience/shortcut module'''
        msd_t, msd_mean, msd_sdt, msd_stack = self.make_sim_avg_MSD(repetitions=20, trace_npts=100000, pot_ampl=0, pot_tilt=0, plots=1)



    def make_potential_sin(self, x_pot, ampl=2, period=26, tilt=0, plots=False):
        ''' periodic sinusoidal potential, ampl and tilt in KT units
            The barrier disappear at a tilt = ampl*pi*period (right?)
        '''
        U = ampl/2*self.KT*np.cos(x_pot*period) - tilt*x_pot/(2*np.pi)*self.KT
        if plots:
            plt.figure('make_potential_sin', clear=True)
            plt.plot(x_pot, U/self.KT, ',')
            plt.ylabel('pot. [KT]')
        return U



    def make_potential_sin2sawtooth(self, x_pot, m, sawint=1, tilt=0, plots=False):
        ''' TODO  make a tilted potential that morphs from a sawtooth(2*x_pot) to a sinus(2*x_pot), pi periodic, 
                sawint : intensity
                m : symmetry from sawtooth (~10) to ~sinus (~1000), perfect sinus m=0
            TODO    add period (26)
                    make class to make potentials, with .period, .ampl, .tilt
        '''
        # force m integer btw 1 and 100
        m = np.round(np.clip(m, 1, len(self.x_pot)/5)).astype(int)
        # L = number of fourier components to use in building the sawtooth:
        if m == 0: 
            L = 1
        else: 
            L = 200 
        # make x_pot longer to avoid edge effects:
        x_pot_conc = np.append(x_pot-2*np.pi, np.append(x_pot, x_pot+2*np.pi))
        # init and build output: 
        saw = np.zeros(len(x_pot_conc))
        for k in range(1, L+1):
            saw = saw + 1/np.pi*(np.sin(k*x_pot_conc)/k )
        saw = 0.5 - saw
        # gaussian filter: 
        saw = self.gaussian_filter(saw, m)
        # crop x_pot in 0,2pi:
        saw = saw[len(x_pot):2*len(x_pot)]
        saw = self.normalize(saw)*sawint
        saw_tilted = saw - tilt*x_pot
        if plots:
            plt.figure('make_sin_sawtooth', clear=1)
            plt.plot(x_pot, saw_tilted)
            plt.grid(True)
        return saw_tilted



    def normalize(self, U):
        '''normalize U between 0,1'''
        # if U not zero, normalize:
        if U.any():
            Un = (U - np.min(U))/(np.max(U) - np.min(U))
        else:
            Un = np.zeros(len(U))
        return Un



    def gaussian_filter(self, data, gsigma=1, plots=0, gmode='reflect'):
        ''' multi dim gaussian filter of data '''
        import scipy.ndimage.filters
        y = scipy.ndimage.filters.gaussian_filter(data, gsigma, mode='reflect')
        if plots:
            plt.figure('gaussian_filter', clear=1)
            plt.plot(data)
            plt.plot(y)
        return y


    def make_trace(self, U, npts=10000, plots=False):
        '''Langevin simulated trace using potential U 
           ex: 
               pp = tilted_periodic_diffusion.TiltedPeriodicDiffusion()
               x = pp.make_trace(pp.make_potential_sin(pp.x_pot, ampl=1, period=10, tilt=10), npts=100000, plots=1)
        '''
        npts = int(npts)
        t0 = time.time()
        self.npts = npts
        # thermal noise:
        thermnoise = np.sqrt(2*self.D*self.dt)*np.random.randn(npts)
        # space derivative of potential U:
        dUdx = np.diff(U)/(2*np.pi/len(U))
        # initial condition:
        x = np.zeros(npts) 
        # numba make trace:
        x = _make_trace(x, npts, dUdx, self.dt, self.gamma, thermnoise)
        # wrap x in (0,2pi):
        x = np.mod(x, 2*np.pi)
        print(f'make_trace(): Done in {time.time() - t0} s.')
        if plots:
            t0 = time.time()
            if npts>1000000:
                idxs = np.linspace(0, npts, 5000, endpoint=False, dtype=int)
                t = np.linspace(0,npts*self.dt, len(idxs), endpoint=False)
                _x = x[idxs]
                print('1')
            else:
                t = np.arange(npts)*self.dt
                _x = x
            fig = plt.figure('make_trace', clear=True, figsize=(12,4))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            ax1.plot(np.linspace(0, 2*np.pi, len(U)), U/self.KT, ',')
            ax1.set_ylabel('Potential [KT]')
            ax1.set_xlabel('angle (rad)')
            ax1.set_title(f'Ampl={(np.max(U)-np.min(U))/self.KT:.1f} KT(pkpk)    dt={self.dt:.1e} s    gamma={self.gamma:.1e} Nsm    D={self.D:.2f} m2/s', fontsize=8)
            print('2')
            ax2.plot(t, _x, ',')
            ax2.set_ylabel('angle (rad)')
            ax2.set_xlabel('time (s)')
            print('3')
            ax3.hist(_x, 200, orientation='vertical')
            ax3.set_xlabel('angle (rad)')
            ax3.set_ylabel('Occur.')
            ax3.set_xlim([0,2*np.pi])
            print('4')
            ax4.plot(t, np.unwrap(_x)/(2*np.pi), lw=1)
            ax4.set_xlabel('time (s)')
            ax4.set_ylabel('angle (turns)')
            ax4.set_title(f'mean speed: {(np.unwrap(x)[-1] - np.unwrap(x)[0])/t[-1]:.1f} Hz', fontsize=8)
            print(f'make_trace(): plot done in {time.time() - t0} s.')
            print('5')
            fig.tight_layout()
            print('6')
        return x
    
   

    def make_MSD(self, x, jmax=100, jskip=10, plots=False):
        ''' modified from Svoboda Block way of calculate the MSD of one unwrapped time trace
                x : time trace
                jmax, jskip : calculate MSD at times in range(1,jmax,jskip)
        '''
        MSD_idxs = np.arange(1, jmax, jskip)
        MSD = np.zeros(len(MSD_idxs))
        # unwrap angle time trace:
        x_unw = np.unwrap(x)
        for k,j in enumerate(MSD_idxs):
            print(f'make_MSD(): {j}/{jmax}', end='\r')
            MSD[k] = np.mean((x_unw[j:] - x_unw[:-j])**2)
        if plots:
            fig = plt.figure('make_MSD', clear=True)
            ax1 = fig.add_subplot(111)
            ax1.plot(MSD_idxs, MSD, '.')
            fig.tight_layout()
        return MSD_idxs, MSD


    
    def make_sim_avg_MSD(self, repetitions=3, trace_npts=1000, pot_ampl=1, pot_tilt=1, plots=False):
        ''' make several simulations, and find the average of the MSD of each 
                repetitions : numb of repeats, so numb.of MSDs to average together
                trace_npts : len of each trace 
                pot_ampl : potential amplitude in KT for the simulation
                pot_tilt : potential tilt in KT
            TODO make parallel
        '''
        msd_stack = []
        potential = self.make_potential_sin(self.x_pot, ampl=pot_ampl, period=26, tilt=pot_tilt)
        if plots:
            fig = plt.figure('make_sim_avg_MSD', clear=True)
            ax1 = fig.add_subplot(111)
        for i in range(repetitions):
            print(f'make_sim_avg_MSD(): repetition {i}/{repetitions}\r')
            x = self.make_trace(potential, npts=trace_npts, plots=i==0)
            msd_idxs, msd = self.make_MSD(x, jmax=len(x)//5, jskip=int((len(x)//5)/20), plots=False)
            if len(msd_stack)==0:
                msd_stack = msd
            else:
                msd_stack = np.vstack((msd_stack, msd))
            if plots:
                ax1.plot(msd_idxs*self.dt, msd, 'o-', ms=1, lw=1, alpha=0.5)
        # avg and std of all MSDs:
        msd_mean = np.mean(msd_stack, axis=0)
        msd_std  = np.std(msd_stack, axis=0)
        msd_time = msd_idxs*self.dt
        self.fit_msd(msd_time, msd_mean, plots=1)
        if plots:
            ax1.plot(msd_time, msd_mean, 'k--', lw=3, label='mean')
            ax1.fill_between(msd_time, msd_mean-msd_std, msd_mean+msd_std, color='k', lw=0, alpha=0.2, label='std')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('MSD')
            ax1.legend()
            fig.tight_layout()
        return msd_time, msd_mean, msd_std, msd_stack

   

    def fit_msd(self, msd_time, msd, plots=False):
        ''' fit msd(t) with a + bt + ct^alpha '''
        from scipy.optimize import curve_fit
        def alpha_poly(t, a,b,alpha):
            return a*t + b*t**alpha
        eps = 1e-9
        # at + bt^2 :
        p0 = [500, 10000, 2]
        bounds_2 = [(-np.inf, -np.inf, 2-eps), (np.inf, np.inf, 2+eps)]
        popt_2, _ = curve_fit(alpha_poly, msd_time, msd, p0=p0, bounds=bounds_2)
        # at + bt^2 :
        p0 = [500, 0, 0]
        bounds_1 = [(-np.inf, -eps, -eps), (np.inf, eps, eps)]
        popt_1, _ = curve_fit(alpha_poly, msd_time, msd, p0=p0, bounds=bounds_1)
        print(f'fit_msd(): Simulation \t\t D = {self.D:.1f}')
        print(f'fit_msd(): Fit 2Dt + bt^2. \t D = {popt_2[0]/2:.1f}')
        print(f'fit_msd(): Fit 2Dt + bt.   \t D = {popt_1[0]/2:.1f}')
        if plots:
            fig = plt.figure('fit_msd', clear=True)
            ax1 = fig.add_subplot(111)
            ax1.plot(msd_time, msd, 'o')
            ax1.plot(msd_time, alpha_poly(msd_time, *popt_2), '--', lw=2)
            ax1.plot(msd_time, alpha_poly(msd_time, *popt_1), 'r--', lw=2)
            ax1.set_title(f'fit: {popt_2[0]:.1f} *t + {popt_2[1]:.1f} *t^{popt_2[2]:.1f}\n{popt_1[0]:.1f} *t ')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('MSD')
                       
    



