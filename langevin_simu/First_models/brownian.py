import numpy as np
import matplotlib.pyplot as plt


#This script simulate the random walk of particle for different timestep

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)



def random_walk(order, dt):

    t = np.arange(0,30,dt)
    x = np.zeros(len(t))
    sd = np.sqrt(dt) 

    for i in range(1,len(t)):
        w = np.random.normal(0.0,sd)
        x[i] = x[i-1] + np.sqrt(dt)*w
        x[i-1] = x[i]
    
    if order == 1 :
        ax1.set_ylim(-8,8)
        ax1.scatter(t, x, color='blue', s=2)
    if order == 2 :
        ax2.set_ylim(-8,8)
        ax2.scatter(t, x, color='blue', s=2)    
    if order == 3 :
        ax3.set_ylim(-8,8)
        ax3.scatter(t, x, color='blue', s=2)  


random_walk(1,1)
random_walk(2,0.5)
random_walk(3,0.1)

ax1.set_title((r'$\Delta$ t= 1'))
ax1.set_xlabel('t')
ax1.set_ylabel('position')
ax2.set_title(r'$\Delta$ t= 0.5')
ax2.set_xlabel('t')
ax2.set_ylabel('position')
ax3.set_xlabel('t')
ax3.set_ylabel('position')
ax3.set_title(r'$\Delta$ t= 0.1')
plt.show()