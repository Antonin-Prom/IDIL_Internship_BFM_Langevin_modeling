from  Langevin_Class_v3 import *

"""
Plot 1: Free Diffusion Trajectories
"""

free_diff = LangevinSimulator(dt=1e-4, torque = 0, x0 = 0,analytical = False)
free_diff.configure_plots()
ampl = 0
npts = int(1e6)

def make_free_trajs(repetitions):
    x0s = np.zeros(repetitions)
    trajs = free_diff.run_parallel_numba(repetitions = repetitions, npts=npts, x0 = x0s, Amplitude = 0,save=False )
    

def plot_free_trajs(trajs,std):
    trajs = trajs[:20]  # Limit to the first 20 trajectories
    t = np.arange(len(trajs[0])) * 1e-4  # Time array
    plt.figure()
    
    for i in range(len(trajs)):
        plt.plot(t, trajs[i], linewidth=0.5)
    
    plt.fill_between(t, -std, std, color='gray', alpha=0.6)
    plt.xlabel('Time [s]',fontsize = 14)
    plt.ylabel('Displacement [rad]',fontsize = 18)
    
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