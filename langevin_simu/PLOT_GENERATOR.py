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
Tilted periodic potential
drift velocity:
    Quadrant with potential, histogramm or Diffusion coeff compare to force, drift velocity, MSD
"""

"""
Critical tilt
"""

"""
D_eff(F)
"""
j = LangevinSimulator(dt=1e-4)
ampl_arr = np.array([10,15,20,25])/2
colors= viridis(np.linspace(0,1,len(ampl_arr)))
tilt_arr = np.arange(0,10)
all_D_eff = []
for A in ampl_arr:
    D_eff = []
    for tilt in tilt_arr:
        tilt_sim = LangevinSimulator(dt=1e-5,torque=tilt)
        print(tilt_sim.torque)
        D = tilt_sim.lifson_jackson(A)
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



    
    
