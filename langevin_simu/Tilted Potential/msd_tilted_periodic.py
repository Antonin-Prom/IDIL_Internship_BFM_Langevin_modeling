
from Class_particle_v2_tilted_potential import *
import scipy
from scipy.optimize import curve_fit

p = DiffusionSimulation2(dt = 0.0001,torque=0,frequency=10)
"""
traj0 = np.load('trajectories_1000000points_amplitude_5kT_dt_0.0001_torque_0.npy',allow_pickle=True)
time_axis,mean_msd = p.mean_msd_and_time_axis(traj0, n_jobs=-1, time_end=1/4, msd_nbpt = 2000, nb_traj = 40)
time_axis *= p.dt_s
np.save(f'tilted_LJ_t,msd,dt={p.dt_s},{5}kT,tilt={0}',[time_axis,mean_msd])"""


def parabolic_drift(t,D,shift):
    return 2*D*t  + shift #+ (v_eff*v_eff)*t*t 

a = 2*np.pi/26
for tilt in range(0,6):
    if tilt == 0:
        _,mean_msd = np.load(f'tilted_LJ_t,msd,dt=0.0001,5kT,tilt=0.npy',allow_pickle=True)
    else:
        _,mean_msd = np.load(f'tilted_LJ_t,msd,dt=0.0001,5.0kT,tilt={tilt}.npy',allow_pickle=True)
    min_absciss_value = 15  # Adjust this value as needed


    
    max_lagtime = int(1000000 * 1/4)
    msd_nbpt = 2000
    time_axis = np.concatenate(([0],np.unique((np.floor(np.logspace(0, (np.log10(max_lagtime)), msd_nbpt)))))) 
    time_axis *= p.dt_s

    absciss = time_axis
    window_indices = np.where(absciss > min_absciss_value)[0]
    absciss_window = absciss[window_indices]
    norm_msd_window = mean_msd[window_indices]
    real_tilt = tilt*p.k_b*p.T_K
    popt, pcov = scipy.optimize.curve_fit(parabolic_drift, absciss_window, norm_msd_window,)
    print("LAAAAAAAAA",popt)
    D_fit, shift = popt[0],popt[1]
    
    print('D_fit = ',D_fit)
    p = DiffusionSimulation2(dt = 0.0001,torque=tilt,frequency=10)
    D_eff = p.lifson_jackson_noforce(5)
    print('D_eff = ',D_eff)

    frac = 2

    mean_msd *= 1/(a*a)
    time_axis *= 1#p.rotational_einstein_diff/(a*a)
    plt.plot(absciss[int(len(absciss)/frac):],parabolic_drift(absciss,D_eff,0.1)[int(len(absciss)/frac):],linestyle ='--',color='black')
    plt.plot(time_axis,mean_msd, label = f'Torque = {tilt}kT')
plt.xlabel('Dt/L²')
plt.ylabel(r'<$\theta$>²/L²')
plt.legend()
plt.show()
    