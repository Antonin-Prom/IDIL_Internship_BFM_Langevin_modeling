from  Langevin_Class_v3 import *
from  Histogram_all_v2 import *

"""
Plot 1: Free Diffusion Trajectories
"""
free_diff = LangevinSimulator(dt=1e-4, torque = 0, x0 = 0,analytical = False)
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
Plot 4: Free Diffusion Histogram
"""


def matrix_at_t(trajs,t):
    m = []
    for traj in trajs:
        traj=np.unwrap(traj)
        m.append(traj[t])
    return m
    
def plot_histograms(npts=1e6, repetitions=1000, free=False, drift=False, periodic=False, tilted_periodic=False, load=False, file_name=None):
    x0 = np.ones(repetitions) * np.pi
    time_select = [1 * int(npts / 10), 4 * int(npts / 10), 7 * int(npts / 10), int(npts - 1)]
    
    if free:
        free = LangevinSimulator(dt=1e-4)
        if not load:
            t0 = time.time()
            trajs = free.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts=int(npts), x0=x0, Amplitude=0, torque=0, iteration=0, save=False, print_time=False, plots=False)
            print(f'Generate done in {time.time() - t0:.1f} s')
        else:
            t1 = time.time()
            trajs = np.load(f'{file_name}')
        matrixes = [matrix_at_t(trajs, t) for t in time_select]
        print(f'Load done in {time.time() - t1:.1f} s')
    
    if drift:
        drift = LangevinSimulator(dt=1e-4, torque=10)
        if not load:
            t0 = time.time()
            trajs = drift.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts=int(npts), x0=x0, Amplitude=0, torque=0, iteration=0, save=False, print_time=False, plots=False)
            print(f'Generate done in {time.time() - t0:.1f} s')
        else:
            t1 = time.time()
            trajs = np.load(f'{file_name}')
        trajs = np.unwrap(trajs)
        matrixes = [matrix_at_t(trajs, t) for t in time_select]
        print(f'Load done in {time.time() - t1:.1f} s')
    
    if periodic:
        periodic = LangevinSimulator(dt=1e-4, torque=0)
        if not load:
            trajs = periodic.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts=int(npts), x0=x0, Amplitude=1, torque=0, iteration=0, save=False, print_time=False, plots=False)
        else:
            trajs = np.load(f'{file_name}')
        trajs = np.unwrap(trajs)
        matrixes = [matrix_at_t(trajs, t) for t in time_select]
    
    if tilted_periodic:
        tilted_periodic = LangevinSimulator(dt=1e-4, torque=10)
        if not load:
            trajs = tilted_periodic.run_parallel_numba(repetitions=repetitions, n_jobs=5, npts=int(npts), x0=x0, Amplitude=4, torque=0, iteration=0, save=False, print_time=False, plots=False)
        else:
            trajs = np.load(f'{file_name}')
        trajs = np.unwrap(trajs)
        matrixes = [matrix_at_t(trajs, t) for t in time_select]
    
    all_data = np.concatenate(matrixes)
    x_min, x_max = np.min(all_data), np.max(all_data)
    y_min, y_max = 0, max([np.histogram(m, bins=repetitions)[0].max() for m in matrixes])
    
    # Plotting histograms in quadrants
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    labels = [f't = {time_select[0] * 1e-4}s', f't = {time_select[1] * 1e-4}s', f't = {time_select[2] * 1e-4}s', f't = {time_select[3] * 1e-4}s']
    axs = axs.flatten()
    
    colors = viridis(np.linspace(0, 1, 4))
    
    for idx, m in enumerate(matrixes):
        axs[idx].hist(m, bins=int(repetitions), alpha=0.7, density=True, color=colors[idx])
        (mu, sigma) = norm.fit(m)
        xmin, xmax = axs[idx].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        axs[idx].plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
        axs[idx].set_title(f'Time {labels[idx]}', fontsize=label_fontsize)
        axs[idx].set_xlim(x_min, x_max)
        axs[idx].set_ylim(y_min, y_max)
        axs[idx].set_xlabel('Position', fontsize=label_fontsize)
        axs[idx].set_ylabel('Frequency', fontsize=label_fontsize)
        axs[idx].legend(fontsize=legend_fontsize)
    
    plt.tight_layout()
    plt.savefig('free_histograms.png', dpi=300)
    plt.show()

plot_histograms(free=True, load=False, file_name='0ite_trajectories_1000000,nb_traj_1000points_amplitude_0kT,frequency_10_dt_0.0001_torque_0kT.npy')









"""
Tilted periodic potential
drift velocity:
    Quadrant with potential, histogramm or Diffusion coeff compare to force, drift velocity, MSD
"""









    
    
