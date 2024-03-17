import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

# Define the Lorenz system
def lorenz(x, y, z, s, r, b):
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot

# Set initial conditions and time step
dt = 0.01
num_steps = 10000

# Create a figure and axis for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set initial parameter values
sigma = 10
rho = 28
beta = 8 / 3
x_init, y_init, z_init = 0.0, 1.0, 1.05

# Function to update the plot
def update(val):
    # Get the current values of the sliders and text boxes
    s = sigma_slider.val
    r = rho_slider.val
    b = beta_slider.val
    x0 = float(x_text_box.text)
    y0 = float(y_text_box.text)
    z0 = float(z_text_box.text)

    # Initialize arrays to store the trajectory
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = x0, y0, z0

    # Generate the Lorenz fractal with the updated parameters
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    # Update the plot with the new trajectory
    ax.clear()
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Fractal")
    plt.draw()

# Create sliders for adjusting parameters
ax_sigma = plt.axes([0.2, 0.02, 0.65, 0.03])
sigma_slider = Slider(ax_sigma, 'Sigma', 0, 20, valinit=sigma)

ax_rho = plt.axes([0.2, 0.06, 0.65, 0.03])
rho_slider = Slider(ax_rho, 'Rho', 0, 50, valinit=rho)

ax_beta = plt.axes([0.2, 0.10, 0.65, 0.03])
beta_slider = Slider(ax_beta, 'Beta', 0, 5, valinit=beta)

# Create text boxes for setting initial values
ax_x_init = plt.axes([0.2, 0.14, 0.15, 0.05])
x_text_box = TextBox(ax_x_init, 'X0', initial=str(x_init))

ax_y_init = plt.axes([0.45, 0.14, 0.15, 0.05])
y_text_box = TextBox(ax_y_init, 'Y0', initial=str(y_init))

ax_z_init = plt.axes([0.7, 0.14, 0.15, 0.05])
z_text_box = TextBox(ax_z_init, 'Z0', initial=str(z_init))

# Call the update function when the slider value is changed or text box is submitted
sigma_slider.on_changed(update)
rho_slider.on_changed(update)
beta_slider.on_changed(update)
x_text_box.on_submit(update)
y_text_box.on_submit(update)
z_text_box.on_submit(update)

# Generate the initial plot
update(None)

plt.show()
