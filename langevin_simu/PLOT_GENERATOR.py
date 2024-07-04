from  Langevin_Class_v3 import *

free_diff = LangevinSimulator(dt=1e-4, torque = 0, x0 = 0,analytical = False)

"""
Plot 1: Free Diffusion Trajectories
"""


free_diff.configure_plots()

label_fontsize = 18
legend_fontsize=14
viridis = plt.cm.viridis

def Free_diffusion():

    ampl = 0
    npts = int(1e7)
    
    def make_free_trajs(repetitions):
        x0s = np.zeros(repetitions)
        trajs = free_diff.run_parallel_numba(repetitions = repetitions, npts=npts, x0 = x0s, Amplitude = 0,save=False )
        trajs = np.unwrap(trajs)
        np.save(f'free_trajs,dt={free_diff.dt},rep = {repetitions}',trajs)
    
    def plot_free_trajs(trajs,std):
        trajs = trajs[:20]  # Limit to the first 20 trajectories
        t = np.arange(len(trajs[0])) * 1e-4  # Time array
        plt.figure()
        
        for i in range(len(trajs)):
            plt.plot(t, trajs[i], linewidth=0.5)
        
        plt.fill_between(t, -std, std, color='gray', alpha=0.6)
        plt.xlabel('Time [s]',fontsize = label_fontsize)
        plt.ylabel('Displacement [rad]',fontsize = label_fontsize)
        
        plt.grid(True)
        plt.xlim(0,100)
        plt.tight_layout()
        plt.savefig('free_trajs_plot.png', dpi=300)
        plt.show()
        
    """
    #trajs = np.unwrap(np.load('0ite_trajectories_1000000,nb_traj_1000points_amplitude_0kT,frequency_10_dt_0.0001_torque_0kT.npy'))
    std = np.std(trajs, axis=0)
    plot_free_trajs(trajs,std)
    """        
        
        
"""
Plot 2: Free Diffusion standrad deviation
"""        
        
def plot_different_std():
  
    trajs5 = np.load(f'free_trajs,dt=1e-05,rep = 10000.npy')
    std5 = np.std(trajs5, axis=0)
    
    
    trajs4 = np.load('free_trajs,dt=0.0001,rep = 10000.npy')
    std4 = np.std(trajs4, axis=0)
    
    
    trajs3 = np.load('free_trajs,dt=0.001,rep = 10000.npy')
    std3 = np.std(trajs3, axis=0) 
    t = np.arange(len(trajs3[0]))*1e-3
    
    
    colors = viridis(np.linspace(0, 1, 3))   
    plt.figure()    
    plt.plot(t,std5[::100],color=colors[0],label= r'dt = 10 $\mu$s')
    plt.plot(t,std4[::10],color=colors[1],label= r'dt = 100 $\mu$s')
    plt.plot(t,std3,color=colors[2],label= r'dt = 1000 $\mu$s')
    plt.plot(t,np.sqrt(2*free_diff.D*t),color='r',linestyle='--',linewidth=3,label=r'$\sqrt{2Dt}$')
    plt.legend(fontsize = legend_fontsize)
    plt.grid(True)
    plt.xlabel('Time [s]',fontsize = label_fontsize)
    plt.ylabel('Standard deviation [rad]',fontsize = label_fontsize)
    plt.xlim(0,1)
      
    plt.tight_layout()
    plt.savefig('free_std_plot1.png', dpi=300)
    plt.show()

    
    
"""
Plot 3: Free Diffusion mean square displacement
"""        
    
def linear_diff(t,D):
    return 2*D*t

def free_MSD():
    colors = viridis(np.linspace(0, 1, 10)) 
    trajs = np.load('free_trajs,dt=0.0001,rep = 10000.npy')
    time_axis,mean_msd_d,matrix = free_diff.mean_msd_and_time(trajs, n_jobs=5, time_end=1/4, msd_nbpt = 500, nb_traj = None)
    t = time_axis * free_diff.dt
    std = np.std(matrix,axis=0)
    
    popt, pcov = scipy.optimize.curve_fit(linear_diff, t[:-1], mean_msd_d[:-1])#, sigma=std, absolute_sigma=True) Less accurate with force fit ???
    residuals = (mean_msd_d - linear_diff(t, *popt))
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((mean_msd_d - np.mean(mean_msd_d))**2)
    R_squared = 1 - (SS_res / SS_tot)
    print(R_squared)
    print('Numerical D = ', popt[0] )
    print('Theoretical D = ', free_diff.D)
    
    # Plotting the results
    plt.figure()
    plt.plot(t, mean_msd_d, label='Numerical MSD', color=colors[0],linewidth=4)
    plt.plot(t, linear_diff(t, *popt), linestyle='--', color='r', label=f'Fitted Curve')
    plt.fill_between(t[:-1], mean_msd_d[:-1] - std, mean_msd_d[:-1] + std, color=colors[4], alpha=0.4)
    plt.xlim(0,0.24)
    plt.xlabel('Lag time (s)',fontsize=label_fontsize)
    plt.ylabel('MSD (rad²)',fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig('free_msd.png', dpi=300)
    plt.show()


""" 
Periodic MSD
"""
def periodic_msd():
    p = LangevinSimulator(dt=1e-4, frequency=10)
    a = 2 * np.pi / p.frequency
    
    def linear_msd(t, D):
        return 2 * D * t
    
    def defaveri_stationary(A):
        X = np.linspace(0, a, 1000)
        Y = boltz_exp(A, X)
        Z = np.mean(Y)
        return np.mean((X**2) * Y) / Z
    
    def boltz_exp(A, x):
        return np.exp(-(A / 2) * np.sin(x * p.frequency))
    
    t_box, msd_box = np.load('langevin_simu\A=0,4,6,18_t,msd_10000000npts_30rep_torque_0kT_dt=0.0001,bead.npy')
    A_box = [0, 4, 6, 18]
    colors = viridis(np.linspace(0, 1, len(A_box)))
    
    plt.figure()
    for idx, (t, msd) in enumerate(zip(t_box, msd_box)):
        print(idx)
        t *= 1e-4
        color = colors[idx]
        if idx < 1:
            min_absciss_value = 0
        else:
            min_absciss_value = 10
        window_indices = np.where(t > min_absciss_value)[0]
        D_eff = p.lifson_jackson(A_box[idx])
        
        label = fr'$V_0 = {A_box[idx]} \, kT, \, D_{{eff}}/D_r = {D_eff / p.D:.2f}$'   
        plt.loglog(t, msd, label=label, color=color)
        if idx<3:
            plt.loglog(t[window_indices], linear_msd(t, D_eff)[window_indices], linestyle='--', color='black')
        else:
            quasi_stat = defaveri_stationary(A_box[idx])
            plt.loglog(t,np.ones(len(t))*quasi_stat,color = 'red',linestyle='--')
    plt.xlabel('Lag time (s)', fontsize=label_fontsize)
    plt.ylabel('MSD (rad²)', fontsize=label_fontsize)
    plt.legend()
    plt.tight_layout()
    plt.savefig('periodic_msd.png', dpi=300)
    plt.show()

"""
Tilted periodic potential
drift velocity:
    Quadrant with potential, histogramm or Diffusion coeff compare to force, drift velocity, MSD
"""

def msd_periodic_tilted():

    def parabolic_msd(t, D, v_eff):
        return 2 * D * t + v_eff*t**2
    def linear_msd(t, D):
        return 2 * D * t

    def boltz_exp(A, x):
        return np.exp(-(A / 2) * np.sin(x * p.frequency))

    #t_box0, msd_box0 = np.load('langevin_simu\A=0,4,6,18_t,msd_10000000npts_30rep_torque_0kT_dt=0.0001,bead.npy')
    t_box, msd_box10 = np.load('langevin_simu\A=0,4,6,18_t,msd_10000000npts_30rep_torque_10kT_dt=1e-05,bead.npy')
    t_box, msd_box50 = np.load('langevin_simu\A=0,4,6,18_t,msd_10000000npts_30rep_torque_50kT_dt=1e-05,bead.npy')
    t_box, msd_box100 = np.load('langevin_simu\A=0,4,6,18_t,msd_10000000npts_30rep_torque_100kT_dt=1e-05,bead.npy')

    A_box = [0, 4, 6, 18]
    tilt_box = [10,50,100]
    t_boxbox = [t_box,t_box,t_box]
    #Only A = 6KT:
    j=3
    msd_boxbox = [msd_box10[j],msd_box50[j],msd_box100[j]]
    colors = viridis(np.linspace(0, 1, len(A_box)))
    t = t_box[0]
    t *= 1e-5
    plt.figure()

    for idt,tilt in enumerate(tilt_box):
        
        msd = msd_boxbox[idt]
        color = colors[idt]
        p = LangevinSimulator(dt=1e-4, frequency=10,torque=tilt)
        F_c =  p.frequency * A_box[j]/2
        print('F_c',F_c)
        if j < 1:
            min_absciss_value = 0
        else:
            min_absciss_value = 10
        window_indices = np.where(t > min_absciss_value)[0]
        popt, _ = scipy.optimize.curve_fit(parabolic_msd, t, msd)
        msd_no_parabolic = msd - popt[1]*t**2
        popt, _ = scipy.optimize.curve_fit(parabolic_msd, t, msd_no_parabolic)
        print('corrected_veff',popt[1])
        D_f, _ = scipy.optimize.curve_fit(linear_msd, t, msd_no_parabolic)
        D_eff = p.lifson_jackson(A_box[j])
        print('D_f,D_eff',D_f/p.D)
        label = fr'$V_0 = {A_box[j]}kT \, F = {tilt/(2*np.pi) :.0f}kT$'   
        #plt.loglog(t, msd, label=label, color=color)
        plt.loglog(t, msd, label=label, color=color)

        #if j<3:
            #plt.loglog(t[window_indices], linear_msd(t, D_eff)[window_indices], linestyle='--', color='black')

    plt.xlabel('Lag time (s)', fontsize=label_fontsize)
    plt.ylabel('MSD (rad²)', fontsize=label_fontsize)
    plt.legend()
    plt.tight_layout()
    plt.savefig('periodic_msd.png', dpi=300)
    plt.show()

"""
Critical tilt
"""

# Define the potential U(x). Example: A sinusoidal potential
def ReimanD_eff(particle,A,F):
    def U(x):
        return particle.analytical_potential(x,A)
    # Constants (you should define these according to your problem)
    L = 2 * np.pi/particle.frequency  # Period of the potential
    D0 = particle.D  # Free diffusion coefficient


    # Functions I_+(x) and I_-(x)
    def I_plus(x):
        return quad(lambda y: np.exp((U(x) - U(y) - F * (x - y)) ), 0, L)[0] / D0

    def I_minus(x):
        return quad(lambda y: np.exp(-(U(x) - U(y) - F * (x - y)) ), 0, L)[0] / D0

    # Compute the integrals I_+ and I_-
    x_values = np.linspace(0, L, 100)
    I_plus_values = [I_plus(x) for x in x_values]
    I_minus_values = [I_minus(x) for x in x_values]

    # Numerically integrate I_+ and I_-
    I_plus_integral = np.trapz(I_plus_values, x_values) / L
    I_minus_integral = np.trapz(I_minus_values, x_values) / L

    # Compute D_eff using equation (8)
    D_eff = D0 * I_minus_integral / (I_plus_integral ** 3)
    return D_eff

"""
Simulation parameters
"""
frequency = 3
A = 30
F_c = frequency*A*np.pi
F_box = np.linspace(0.5*F_c,2*F_c,150)
dt = 1e-5
N =int(1e7)
repetitions = 5
def generate_msd_f_box():
    for F in F_box:
        p = LangevinSimulator(frequency=frequency,dt=dt,torque=F)
        p.brutal_msd(repetition=repetitions,N=N,Amplitude=A,id=F_c)
#generate_msd_f_box()

#F_box = np.concatenate([np.linspace(0, 2*F_c, 15), np.linspace(2*F_c, 6*F_c, 30)])
#F_box = np.linspace(0, 2*F_c, 15)

def parabolic_msd(t, D, v_eff):
    return 2 * D * t + v_eff*t**2
def linear_msd(t, D):
    return 2 * D * t

#[t_box,msd_box] = [np.load(f'langevin_simu\\t,msd_10000000npts_20rep_torque_{F}kT_A=5.0,dt=1e-05,bead.npy') for F in F_box]

msd_box=[]
for F in F_box:
    
    file_path = f't,msd_10000000npts_5rep_torque_{F}kT_dt=1e-05,id_{F_c}_bead,removed_mean.npy'
    t, msd = np.load(file_path)
    msd_box.append(msd)

t*=dt
D_fit_box = []
D_theo_box = []
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for id,msd in enumerate(msd_box):
    F = F_box[id]
    p = LangevinSimulator(dt=dt, frequency=frequency,torque=F)
    #D_theo = p.lifson_jackson(A)
    #D_theo_box.append(D_theo)
    D_f, _ = scipy.optimize.curve_fit(linear_msd, t, msd)
    D_fit_box.append(D_f/p.D)
    ax1.loglog(t, msd)#label=f'F = {F/(2*np.pi):.0f}kT.$rad^-1$'
    
# Plot F vs D_fit
ax2.plot(F_box, D_fit_box, 'o-')
#ax2.plot(F_box, D_theo_box, 'o-')
ax2.set_xlabel('F[kT.$rad^{-1}$]')
ax2.set_ylabel('$D_{fit}$')
print(D_theo_box)
# Customize the subplots
ax1.set_xlabel('Lag time [s]')
ax1.set_ylabel('rad²')
ax1.legend()
ax1.set_title(r'Var($\theta$)')

ax2.axvline(F_c, color='red', linestyle='--', linewidth=1)
ax2.annotate('$F_c$', xy=(F_c, max(D_fit_box)), xytext=(F_c, max(D_fit_box)*1.1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center', verticalalignment='bottom')

plt.tight_layout()
plt.savefig('D_fit(F).png', dpi=300)
plt.show()

"""
D_eff(F)
"""
def D_eff_f_theo():
    j = LangevinSimulator(dt=1e-4)
    ampl_arr = np.array([10,15,20,25])/2
    colors= viridis(np.linspace(0,1,len(ampl_arr)))
    tilt_arr = np.arange(0,100)
    all_D_eff = []
    for A in ampl_arr:
        D_eff = []
        for tilt in tilt_arr:
            tilt *= -1
            tilt_sim = LangevinSimulator(dt=1e-5,torque=tilt)
            D = tilt_sim.D_eff_reimann(A)
            D_eff.append(D/j.D)
        all_D_eff.append(D_eff)
        
    plt.figure()        
    for idx,A in enumerate(ampl_arr):
        plt.plot(tilt_arr, all_D_eff[idx], label='Numerical MSD', color=colors[idx],linewidth=4)
        
    plt.xlabel('F',fontsize=label_fontsize)
    plt.ylabel('D_eff',fontsize=label_fontsize)
    plt.tight_layout()
    plt.savefig('D_eff(F)', dpi=300)
    plt.show()

def D_eff(F):
    j = LangevinSimulator(dt=1e-4,frequency=10)
    ampl_arr = np.array([5])
    colors= viridis(np.linspace(0,1,len(ampl_arr)))


    all_D_eff = []
    for A in tqdm(ampl_arr, desc="Amplitude loop"):
        F_c = 2 * np.pi * j.frequency * A
        tilt_arr = np.linspace(F_c / 2, 1.5 * F_c, 5)
        
        D_eff = []
        D_theo = []
        for tilt in tqdm(tilt_arr, desc=f"Tilt loop for A={A}", leave=False):
            d = LangevinSimulator(dt=1e-5, torque=tilt, frequency=10)
            time_axis, mean_msd = d.brutal_msd(repetition=5, N=int(1e7), Amplitude=A, x0=[0], ide=0, msd_nbpt=500, time_end=1/4, save=False)
            time_axis *= d.dt
            v_eff = tilt * j.KT / j.gamma
            mean_msd_no_parabolic = [mean_msd[j] - (time_axis[j]**2 * v_eff**2) for j,_ in enumerate(mean_msd)]
            
            plt.loglog(time_axis,mean_msd,color='salmon')
            plt.loglog(time_axis,mean_msd_no_parabolic,color='blue')
            popt, _ = scipy.optimize.curve_fit(linear_msd, time_axis, mean_msd_no_parabolic)
            D_fit = popt[0]
            D_eff.append(D_fit)
            D_theo.append(d.D_eff_reimann(A))
        all_D_eff.append(D_eff)
        np.save(f'D_eff(F)_A={A}', [tilt_arr, D_eff])
        #plt.plot(tilt_arr, D_eff,color='green')
        #plt.plot(tilt_arr, D_theo, color='blue')
        plt.show()
    
"""
A = 10 / 2
time_axis *= d.dt
a = 2 * np.pi / d.frequency
D_eff = d.lifson_jackson(A)

# Viridis colors
viridis = plt.cm.viridis
colors = viridis(np.linspace(0, 1, 100))

# Functions
def linear_msd(t, D):
    return 2 * D * t

def boltz_exp(A, x):
    return np.exp(-d.analytical_potential(x, A))

def defaveri_stationary(A):
    X = np.linspace(-a / 6, a / 6, 1000)
    Y = boltz_exp(A, X)
    Z = np.mean(Y)
    return np.mean((X ** 2) * Y) / Z

# Calculation
min_absciss_value = 1
window_indices = np.where(time_axis > min_absciss_value)[0]
qstat = defaveri_stationary(A)
equi = np.ones(len(time_axis)) * qstat

# Fit
popt, _ = scipy.optimize.curve_fit(linear_msd, time_axis[window_indices], mean_msd[window_indices])
D_fit = popt[0]
"""

    
    
