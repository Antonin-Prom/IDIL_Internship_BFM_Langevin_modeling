from  Langevin_Class_v3 import *

"""
Plot 1: Free Diffusion Trajectories
"""

free_diff = LangevinSimulator(dt=1e-4, torque = 0, x0 = 0,analytical = False)
free_diff.configure_plots()
ampl = 0
npts = int(1e7)

def make_free_trajs(repetitions):
    x0s = np.zeros(repetitions)
    trajs = free_diff.run_parallel_numba(repetitions = repetitions, npts=npts, x0 = x0s, Amplitude = 0 )
    std = np.std(trajs, axis=0)
    return trajs,std

def plot_free_trajs(trajs,std):
    trajs = trajs[:20]  # Limit to the first 20 trajectories
    t = np.arange(len(trajs[0])) * 1e-4  # Time array
    plt.figure()
    
    for i in range(len(trajs)):
        plt.plot(t, trajs[i], linewidth=0.5)
    
    plt.fill_between(t, -std, std, color='gray', alpha=0.6, label='std')
    
    plt.title('Angular Trajectories of Freely Diffusing Particles')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [rad]')
    
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('free_trajs_plot.png', dpi=300)
    plt.show()

trajs,std = make_free_trajs(1000)
plot_free_trajs(trajs,std)