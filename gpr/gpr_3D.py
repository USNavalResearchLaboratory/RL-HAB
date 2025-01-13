import numpy as np
import time
from env.config.env_config import env_params
import matplotlib.pyplot as plt
from utils.initialize_forecast import initialize_forecasts
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from termcolor import colored
import os
from joblib import load, dump




#Forecast processing
env_params["timewarp"] = None
FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

#env_params["rel_dist"] = 100_000_000  # Manually Override relative distance to show a whole subset

timestamp = "2023-07-17T00:00:00.000000000"

forecast_subset_synth.assign_coord(
    0.5 * (forecast_subset_synth.Forecast.LAT_MAX + forecast_subset_synth.Forecast.LAT_MIN),
    0.5 * (forecast_subset_synth.Forecast.LON_MAX + forecast_subset_synth.Forecast.LON_MIN),
    timestamp)
forecast_subset_synth.subset_forecast(days=1)

forecast_subset_era5.assign_coord(forecast_subset_synth.lat_central, forecast_subset_synth.lon_central, timestamp)
forecast_subset_era5.subset_forecast(days=1)

#New Era 5 slice stuff
era5_slice = copy.deepcopy(forecast_subset_era5)
era5_slice.ds = era5_slice.ds.sel(time=timestamp)

#Why do we unreverse latitude from forecast processing plotting now??????????????????????????????????????????
#Is this because of the Transpose step in the other opt
era5_slice.ds = era5_slice.ds.reindex(latitude=list(reversed(era5_slice.ds.latitude)))

ERA5_u_values = era5_slice.ds['u'].values.squeeze()
ERA5_v_values = era5_slice.ds['v'].values.squeeze()

#Reshape data from (lev, lat lon) to (lat, lon, lev)
ERA5_u_values = ERA5_u_values.transpose(1, 2, 0)
ERA5_v_values = ERA5_v_values.transpose(1, 2, 0)

#Determine gridding
y = era5_slice.ds.latitude.values
x = era5_slice.ds.longitude.values
z = era5_slice.ds.level.values

# Step 2: Create a meshgrid of X, Y, Z
y_grid, x_grid, z_grid = np.meshgrid(y, x, z, indexing="ij")  # Shape: (50, 30, 20)

# Choose a color data function
#color_data = ERA5_u_values

# Step 4: Flatten the grid and color data for GPR
x_flat = x_grid.ravel()
y_flat = y_grid.ravel()
z_flat = z_grid.ravel()
u_flat = ERA5_u_values.ravel()
v_flat = ERA5_v_values.ravel()

print("ERA5 U shape", ERA5_u_values.shape, "FLATTENED", u_flat.shape)


# Step 5: Sample N random points from the flattened grid for GPR
n_sample_points = 100


indices = np.random.choice(len(x_flat), n_sample_points, replace=False)
x_sample = x_flat[indices]
y_sample = y_flat[indices]
z_sample = z_flat[indices]
u_sample = u_flat[indices]
v_sample = v_flat[indices]

# Step 6: Fit Gaussian Process Regression model
train_data = np.column_stack((x_sample, y_sample, z_sample))
print("Train Data shape", train_data.shape)
# Define an anisotropic kernel with different length scales for X, Y, Z
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[5.0, 5.0, 10.0], length_scale_bounds=(1e-2, 1e3))
gpr_u = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr_v = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)




#NEW SAVING AND LOADING
if os.path.exists("gpr_u_model_era5.joblib") and os.path.exists("gpr_v_model_era5.joblib"):
    gpr_u = load("gpr_u_model_era5.joblib")
    gpr_v = load("gpr_v_model_era5.joblib")
    print(colored("GPR models loaded successfully.", "green"))

else:
    print("No saved models found. Refitting GPR models...")

    print("Training GPRs")
    # Measure time for U-Wind GPR fitting
    start_time_u = time.time()
    gpr_u.fit(train_data, u_sample)
    end_time_u = time.time()
    print(colored(f"Finished fitting U-Wind GPR in {end_time_u - start_time_u:.2f} seconds", "green"))

    # Measure time for V-Wind GPR fitting
    start_time_v = time.time()
    gpr_v.fit(train_data, v_sample)
    end_time_v = time.time()
    print(colored(f"Finished fitting V-Wind GPR in {end_time_v - start_time_v:.2f} seconds", "green"))

    # Save the models
    dump(gpr_u, "gpr_u_model_era5.joblib")
    dump(gpr_v, "gpr_v_model_era5.joblib")
    print(colored("GPR models saved successfully.","cyan"))

'''
# Step 4: Generate a new dense grid for prediction
grid_size = 30  # Resolution of the new grid
x_pred = np.linspace(x_grid.min(), x_grid.max(), grid_size)
y_pred = np.linspace(y_grid.min(), y_grid.max(), grid_size)
z_pred = np.linspace(z_grid.min(), z_grid.max(), grid_size)
x_grid, y_grid, z_grid = np.meshgrid(x_pred, y_pred, z_pred)
'''

x_pred = era5_slice.ds.latitude.values
y_pred = era5_slice.ds.longitude.values
z_pred = era5_slice.ds.level.values

# Flatten grid to predict color values
grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
print("Grid Points shape", grid_points.shape)
print("Expected total grid points:", y_grid.size)  # Compare this with `grid_points.shape[0]`

u_pred, u_std = gpr_u.predict(grid_points, return_std=True)
v_pred, v_std = gpr_v.predict(grid_points, return_std=True)


'''
print("Timing GPR example")
start_time_v = time.time()
u_pred, u_std = gpr_u.predict(np.random.rand(1155, 3), return_std=True)
print(u_pred)
end_time_v = time.time()
print(colored(f"Finished fitting V-Wind GPR in {end_time_v - start_time_v} seconds", "green"))
'''

#sdfsdfsd

# Reshape predictions to grid shape
#u_pred = u_pred.reshape(grid_size, grid_size, grid_size)
#v_pred = v_pred.reshape(grid_size, grid_size, grid_size)

u_pred = u_pred.reshape(y_grid.shape)
v_pred = v_pred.reshape(y_grid.shape)
u_std = u_std.reshape(y_grid.shape)
v_std = v_std.reshape(y_grid.shape)

# Step 5: Plot the original sampled points
fig1 = plt.figure(figsize=(6, 4))
ax1 = fig1.add_subplot(111, projection='3d')

# Original sampled points
#sc1 = ax1.scatter(x_sample, y_sample, z_sample, c=color_sample, cmap='jet', marker='o', s=10, alpha=0.7)
sc1 = ax1.scatter(x_flat, y_flat, z_flat, c=v_flat, cmap='jet', s=5, alpha=0.7)

# Colorbar
plt.colorbar(sc1, ax=ax1, shrink=0.5, aspect=10, label="Sampled Color Values")

# Labels and title
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Original Random Sampled Points with Colors')

# Step 6: Plot the GPR-predicted grid
fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(111, projection='3d')

# GPR Predicted Points
sc2 = ax2.scatter(x_grid, y_grid, z_grid, c=v_pred.ravel(), cmap='jet', marker='.', s=10, alpha=0.7)

# Colorbar
plt.colorbar(sc2, ax=ax2, shrink=0.5, aspect=10, label="Predicted Color Values (GPR)")

# Labels and title
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('GPR-Predicted Color Values on 3D Grid')

fig3 = plt.figure(figsize=(6, 4))
ax3 = fig3.add_subplot(111, projection='3d')

# GPR Predicted Points
sc3 = ax3.scatter(x_grid, y_grid, z_grid, c=v_std.ravel(), cmap='plasma', marker='.', s=10, alpha=0.7)

# Colorbar
plt.colorbar(sc3, ax=ax3, shrink=0.5, aspect=10, label="Predicted Color Values (GPR)")

# Labels and title
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('GPR-U-STD Color Values on 3D Grid')






#plt.show()

# --- 2D Slice Visualization ---
chosen_level = z[0]  # Specify the pressure level to visualize
level_idx = np.argmin(np.abs(z - chosen_level))

print(level_idx)

# Extract slices

#REMOVED EXTRA TRANSPOSING BY UNREVERSING LATITUDE IN THE BEGINING OF THE SCRIPT
#ERA5_u_values = ERA5_u_values.transpose(1,0,2)
#ERA5_v_values = ERA5_v_values.transpose(1,0,2)

# Flip the longitude axis
#ERA5_u_values = np.flip(ERA5_u_values, axis=0)  # Flip along the longitude axis
#ERA5_v_values = np.flip(ERA5_v_values, axis=0)


for level_idx in range(0,len(z)):
    chosen_level = z[level_idx]  # Specify the pressure level to visualize
    level_idx = np.argmin(np.abs(z - chosen_level))

    print(level_idx)

    ERA5_u_slice = ERA5_u_values[:, :, level_idx]
    ERA5_v_slice = ERA5_v_values[:, :, level_idx]
    GPR_u_slice = u_pred[:, :, level_idx]
    GPR_v_slice = v_pred[:, :, level_idx]

    print(ERA5_u_values.shape, u_pred.shape)

    # Plot 4 subplots
    uv_colormap = 'seismic'


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'2D Slices at Pressure Level {chosen_level} hPa', fontsize=16)

    print("subplot shapes", x.shape, y.shape, ERA5_u_slice.shape)

    x_edges = np.linspace(x.min(), x.max(), len(x) + 1)
    y_edges = np.linspace(y.min(), y.max(), len(y) + 1)

    # Original ERA5 U-Wind
    c1 = axes[0, 0].imshow(ERA5_u_slice,
                            extent=[x.min(), x.max(), y.min(), y.max()],
                           cmap=uv_colormap)
    fig.colorbar(c1, ax=axes[0, 0], label='ERA5 U-Wind (m/s)')
    axes[0, 0].set_title('Original ERA5 U-Wind')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')

    # Original ERA5 V-Wind
    c2 = axes[0, 1].imshow(ERA5_v_slice,
                            extent=[x.min(), x.max(), y.min(), y.max()],
                           cmap=uv_colormap)
    fig.colorbar(c2, ax=axes[0, 1], label='ERA5 V-Wind (m/s)')
    axes[0, 1].set_title('Original ERA5 V-Wind')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')


    # GPR-Predicted U-Wind
    c3 = axes[1, 0].imshow(GPR_u_slice,
                            extent=[x.min(), x.max(), y.min(), y.max()],
                           cmap=uv_colormap)
    fig.colorbar(c3, ax=axes[1, 0], label='GPR U-Wind (m/s)')
    axes[1, 0].set_title('GPR-Predicted U-Wind')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')

    # GPR-Predicted V-Wind
    c4 = axes[1, 1].imshow(GPR_v_slice,
                            extent=[x.min(), x.max(), y.min(), y.max()],
                           cmap=uv_colormap)
    fig.colorbar(c4, ax=axes[1, 1], label='GPR V-Wind (m/s)')
    axes[1, 1].set_title('GPR-Predicted V-Wind')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')


    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
plt.show()