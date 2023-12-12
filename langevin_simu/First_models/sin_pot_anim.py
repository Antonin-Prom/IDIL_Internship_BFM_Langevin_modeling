import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter
# Initialisation of potential:
amplitude = 2
k = 2

# Physical parameters
kT = 2
friction = 0.5
diffusion = kT / friction

# Finite spatial step
d = 0.001

# Timesteps
dt = 0.001

# Standard deviation
sd = np.sqrt(2 * diffusion * dt)

fig, ax = plt.subplots()
x = np.arange(-10, 10, d)

class Particle:
    def __init__(self, x):
        self.x = x
        self.E = energy(amplitude, self.x, k)
        self.scat = ax.scatter(self.x, self.E)

    def update(self):
        dx = -(1 / friction) * dt * ((energy(amplitude, self.x + d, k) - energy(amplitude, self.x, k)) / d)
        dx += np.random.normal(0.0, sd)
        self.x += dx
        self.E = energy(amplitude, self.x, k)
        self.scat.set_offsets((self.x, self.E))

def energy(A, x, k):
    return A * np.cos(k * x)

E = [energy(amplitude, xi, k) for xi in x]
plt.plot(x, E)

part1 = Particle(0)
part2 = Particle(1)
def animate(frame):
    part1.update()
    part2.update()
    return part1.scat,part2.scat

plt.xlabel('x')
plt.ylabel('E')
plt.title('diffusing particles on sinusoidal potential')
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, 2 * np.pi, dt), blit=True, interval=10)
plt.show()
