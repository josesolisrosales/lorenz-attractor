import numpy as np
import matplotlib.pyplot as plt

# Define the Lorenz system
def lorenz(x, y, z, s=10, r=28, b=8/3):
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot

# Set initial conditions and time step
dt = 0.01
num_steps = 10000

# Function to generate Lorenz attractor trajectory
def generate_lorenz_trajectory(x0, y0, z0, num_steps):
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    xs[0], ys[0], zs[0] = x0, y0, z0

    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    
    return xs, ys, zs

# Set initial values
x_init, y_init, z_init = 1.0, 1.0, 1.0
delta = 0.01  # Small difference in initial condition

# Generate two trajectories with slightly different initial conditions
xs1, ys1, zs1 = generate_lorenz_trajectory(x_init, y_init, z_init, num_steps)
xs2, ys2, zs2 = generate_lorenz_trajectory(x_init + delta, y_init, z_init, num_steps)

# Plot the trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(xs1, ys1, zs1, lw=0.5, label='Trajectory 1')
ax.plot(xs2, ys2, zs2, lw=0.5, label='Trajectory 2 (x + 0.01)')

ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor: Sensitivity to Initial Conditions")
ax.legend()

plt.show()
