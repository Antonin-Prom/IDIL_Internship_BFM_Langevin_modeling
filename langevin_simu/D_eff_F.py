from  Langevin_Class_v3 import *


label_fontsize = 18
legend_fontsize=14
viridis = plt.cm.viridis

frequency = 3
A = 50
F_c = frequency*A*np.pi
F_box = np.linspace(0.5*F_c,2*F_c,80)
dt = 5e-6
N =int(5e7)
repetitions = 8

def generate_msd_f_box():
    for F in F_box:
        p = LangevinSimulator(frequency=frequency,dt=dt,torque=F)
        p.brutal_msd(repetition=repetitions,N=N,Amplitude=A,id=F_c)
generate_msd_f_box()

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

for id,msd in enumerate(msd_box):
    F = F_box[id]
    p = LangevinSimulator(dt=dt, frequency=frequency,torque=F)
    #D_theo = p.lifson_jackson(A)
    #D_theo_box.append(D_theo)
    D_f, _ = scipy.optimize.curve_fit(linear_msd, t, msd)
    D_fit_box.append(D_f/p.D)
    D_theo_box.append(p.D_eff_reimann(A)/p.D)

# Plot F vs D_fit
ax1.plot(F_box, D_fit_box, 'o')
ax1.plot(F_box,D_theo_box)

ax1.set_xlabel('F[kT.$rad^{-1}$]', fontsize = label_fontsize)
ax1.set_ylabel('$D_{fit}$', fontsize = label_fontsize)
print(D_theo_box)
# Customize the subplots

X = np.linspace(-np.pi,np.pi,1000)
U = (A/2)*np.sin(X*frequency) - F_c/(2*np.pi)*X
ax2.plot(X,U)
ax2.set_xlabel('rad', fontsize = label_fontsize)
ax2.set_ylabel('U[kT]',fontsize = label_fontsize)
ax2.legend()

ax1.axvline(F_c, color='red', linestyle='--', linewidth=1)
ax1.annotate('$F_c$', xy=(F_c, max(D_fit_box)), xytext=(F_c, max(D_fit_box)*1.1),
            arrowprops=dict(facecolor='black', shrink=0.05),fontsize = label_fontsize,
            horizontalalignment='center', verticalalignment='bottom')

plt.tight_layout()
plt.savefig('D_fit(F)_theory.png', dpi=300)
plt.show()