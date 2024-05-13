
from Class_particle_v2_tilted_potential import *
import scipy
from scipy.optimize import curve_fit

tilt_box = [1,2,3,4,5,10,20]
A_box = np.linspace(0,5,50)
def generate_msd_tilted():
    tilt_box = [1,2,3,4,5,10,20]
    A_box = np.linspace(0,5,50)
    for tilt in tilt_box:
        p = DiffusionSimulation2(dt=0.0001,frequency=10,torque=tilt)
        for A in A_box:
            trajs = p.run_parallel(repetitions=40, n_jobs=-1, npts = 1000000, Amplitude = A)
            time_axis,mean_msd = p.mean_msd_and_time_axis(trajs, n_jobs=-1, time_end=1/4, msd_nbpt = 3000, nb_traj = 40)
            time_axis *= p.dt_s
            np.save(f'tilted_LJ_t,msd,dt={p.dt_s},{A}kT,tilt={tilt}',[time_axis,mean_msd])

def asdefaveri_msd(A,tilt):
    p = DiffusionSimulation2(dt=0.0001,frequency=10,torque=tilt)
    a = 2*np.pi/p.frequency
    D = p.rotational_einstein_diff           
    D_eff = p.lifson_jackson_force(A)
    
    def linear_D(t,D,shift):
        return (2*D*t + shift)
    
    t, msd = np.load(f'tilted_LJ_t,msd,dt={p.dt_s},{A}kT,tilt={tilt}.npy',allow_pickle=True)
    absciss = t 
    popt, pcov = scipy.optimize.curve_fit(linear_D, absciss, msd)
    D_fit, shift = popt[0], popt[1]
    frac = 2
    #plt.plot(absciss,msd,label = f'A = {A} kT')
    #plt.plot(absciss,linear_D(absciss,D_fit,shift),linestyle ='--',color='black') 
    #plt.plot(absciss,linear_D(absciss,D_eff,shift),linestyle ='--',color='red') 

    print('D_fit = ',D_fit)
    print('D_eff = ',D_eff)
    
    return D_fit,D_eff,pcov

tilt = 3
results = []
D_fit_box = []
D_eff_box = []
for A in A_box:
    p = DiffusionSimulation2(dt=0.0001,frequency=10,torque=tilt)
    np.load(f'tilted_LJ_t,msd,dt={p.dt_s},{A}kT,tilt={tilt}.npy',allow_pickle=True)
    D_fit,D_eff,_ = asdefaveri_msd(A,tilt)
    results.append(D_fit,D_eff)
    D_fit_box.append(D_fit)
    D_eff_box.append(D_eff)
plt.scatter(A_box,D_fit_box,color = "salmon")
plt.plot(A_box,D_eff_box,label='LJ')
plt.legend()
plt.show()