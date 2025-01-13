import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Step 1: Define different dimensions for X, Y, Z
x = np.linspace(0, 1, 3)  # X has 50 points
y = np.linspace(0, 1, 3)  # Y has 30 points
z = np.linspace(-1, 1, 7)  # Z has 20 points

# Step 2: Create a meshgrid of X, Y, Z
x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")  # Shape: (50, 30, 20)

# Choose a color data function
color_data = z_grid  # Colors are derived from the Z values
#color_data = np.sin(10 * x) * np.cos(10 * y) + np.sin(5 * z) # Sinusoidal Function
#color_data = np.sqrt((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.0)**2) # Radial Distance Function
#color_data = np.sin(10 * np.sqrt(x_grid**2 + y_grid**2)) + np.cos(5 * z_grid) # Spiral like function
#color_data = np.exp(-((x - 0.5)**2 / 0.1 + (y - 0.5)**2 / 0.1 + (z)**2 / 0.2)) # Gaussian Like Function
#color_data = np.log1p(x * y) + np.abs(np.sin(5 * z)**3) # Logarithmic and Non-linear Combination
#color_data = ((np.floor(5 * x) + np.floor(5 * y) + np.floor(5 * z)) % 2) #Checkerboard pattern

# Step 4: Flatten the grid and color data for GPR
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()
z_flat = z_grid.ravel()
color_flat = color_data.ravel()

print("color shape", color_data.shape)


print(x)
print(color_data.shape)

# Step 5: Sample N random points from the flattened grid for GPR
n_sample_points = 10

indices = np.random.choice(len(x_flat), n_sample_points, replace=False)
x_sample = x_flat[indices]
y_sample = y_flat[indices]
z_sample = z_flat[indices]
color_sample = color_flat[indices]  # Target is the sampled color data

# Step 6: Fit Gaussian Process Regression model
# Inputs: X, Y, Z; Target: Color values
train_data = np.column_stack((x_sample, y_sample, z_sample))
kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

gpr.fit(train_data, color_sample)

# Step 4: Generate a new dense grid for prediction
grid_size = 30  # Resolution of the new grid
x_pred = np.linspace(0, 1, grid_size)
y_pred = np.linspace(0, 1, grid_size)
z_pred = np.linspace(-1, 1, grid_size)
x_grid, y_grid, z_grid = np.meshgrid(x_pred, y_pred, z_pred)

# Flatten grid to predict color values
grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
color_pred, sigma = gpr.predict(grid_points, return_std=True)

# Reshape predictions to grid shape
color_pred = color_pred.reshape(grid_size, grid_size, grid_size)

# Step 5: Plot the original sampled points
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# Original sampled points
#sc1 = ax1.scatter(x_sample, y_sample, z_sample, c=color_sample, cmap='jet', marker='o', s=10, alpha=0.7)
sc1 = ax1.scatter(x_flat, y_flat, z_flat, c=color_flat, cmap='jet', s=5, alpha=0.7)


# Colorbar
plt.colorbar(sc1, ax=ax1, shrink=0.5, aspect=10, label="Sampled Color Values")

# Labels and title
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Original Random Sampled Points with Colors')

# Step 6: Plot the GPR-predicted grid
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# GPR Predicted Points
sc2 = ax2.scatter(x_grid, y_grid, z_grid, c=color_pred.ravel(), cmap='jet', marker='.', s=10, alpha=0.7)

# Colorbar
plt.colorbar(sc2, ax=ax2, shrink=0.5, aspect=10, label="Predicted Color Values (GPR)")

# Labels and title
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('GPR-Predicted Color Values on 3D Grid')

# Show both figures
plt.show()
