import numpy as np
import matplotlib.pyplot as plt

# Function to generate Brownian motion trajectories
def generate_brownian_motion(num_trajectories, num_steps):
    dt = 1.0  # Time step
    sqrt_dt = np.sqrt(dt)

    # Generate random increments for each trajectory and each time step
    increments = np.random.normal(loc=0, scale=sqrt_dt, size=(num_trajectories, num_steps))

    # Calculate cumulative sum along the time axis to get the trajectories
    trajectories = np.cumsum(increments, axis=1)

    return trajectories

# Function to compute mean variance for each time step
def compute_mean_variance(trajectories):
    # Calculate variance for each time step
    variances = np.var(trajectories, axis=0)

    # Calculate mean variance up to each time step
    mean_variances = np.mean(variances, axis=0)

    return mean_variances

# Set the number of trajectories and time steps
num_trajectories = 100
num_steps = 100

# Generate Brownian motion trajectories
trajectories = generate_brownian_motion(num_trajectories, num_steps)

# Compute mean variance for each time step
mean_variances = compute_mean_variance(trajectories)

# Plot the mean variance over time
plt.plot(np.arange(1, num_steps + 1), mean_variances)
plt.xlabel('Time Step')
plt.ylabel('Mean Variance')
plt.title('Mean Variance of 100 Brownian Motion Trajectories')
plt.show()
