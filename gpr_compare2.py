import numpy as np
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast, Forecast_Subset
import matplotlib.pyplot as plt
from utils.initialize_forecast import initialize_forecasts
import matplotlib.colors as mcolors
import pandas as pd
import copy
from PIL import Image
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def gpr_compare():
    env_params["timewarp"] = None
    FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

    env_params["rel_dist"] = 10_000_000  # Manually Override relative distance to show a whole subset

    timestamp = "2023-07-17T00:00:00.000000000"

    forecast_subset_synth.assign_coord(
        0.5 * (forecast_subset_synth.Forecast.LAT_MAX + forecast_subset_synth.Forecast.LAT_MIN),
        0.5 * (forecast_subset_synth.Forecast.LON_MAX + forecast_subset_synth.Forecast.LON_MIN),
        timestamp)
    forecast_subset_synth.subset_forecast(days=1)

    forecast_subset_era5.assign_coord(forecast_subset_synth.lat_central, forecast_subset_synth.lon_central, timestamp)
    forecast_subset_era5.subset_forecast(days=1)

    # FIND ALTITUDE FOR COMPARISON WITH SYNTH WINDS EXAMPLE USAGE
    """Find nearest pressure levels"""
    alt_era5 = forecast_subset_era5.get_alt_from_pressure(30)

    print("alt_era5", alt_era5)

    # For Synthwinds, can Assume every coordinate has the same altitude column, so just take first index
    alt_column = forecast_subset_synth.ds.isel(time=0, latitude=0, longitude=0)['z'].values / 9.81

    print("synth alts", alt_column)
    print("Num SYNTH LEVELS: ", len(alt_column))

    # Find the index of the nearest z value
    nearest_idx = np.argmin(np.abs(alt_column - alt_era5))

    print("synth_idx", nearest_idx,
          forecast_subset_synth.ds.isel(time=0, latitude=0, level=nearest_idx, longitude=0)['z'].values / 9.81)
    print("*****DONE ****")

    skip = 2

    pres = forecast_subset_era5.ds.level.values[3]
    print(pres)
    alt_era5 = forecast_subset_era5.get_alt_from_pressure(pres)
    # For Synthwinds, can Assume every coordinate has the same altitude column, so just take first index
    alt_column = forecast_subset_synth.ds.sel(time=timestamp).isel(latitude=0, longitude=0)['z'].values / 9.81
    # Find the index of the nearest z value
    nearest_idx = np.argmin(np.abs(alt_column - alt_era5))
    print(pres, alt_era5, forecast_subset_synth.ds.sel(time=timestamp).isel(latitude=0, longitude=0)['z'].values / 9.81)

    synth_slice = copy.deepcopy(forecast_subset_synth)
    synth_slice.ds = forecast_subset_synth.ds.isel(level=slice(nearest_idx, nearest_idx + 1))
    synth_slice.ds = synth_slice.ds.sel(time=timestamp)
    # forecast_visualizer_synth = ForecastVisualizer(synth_slice)
    SYNTH_u_values = synth_slice.ds['u'].values.squeeze()
    SYNTH_v_values = synth_slice.ds['v'].values.squeeze()

    SYNTH_angles = np.arctan2(SYNTH_v_values, SYNTH_u_values) * (180 / np.pi)  # Convert to degrees



    #New Era 5 slice stuff
    era5_slice = copy.deepcopy(forecast_subset_era5)
    era5_slice.ds = forecast_subset_era5.ds.sel(level=pres)
    era5_slice.ds = era5_slice.ds.sel(time=timestamp)
    # forecast_visualizer_synth = ForecastVisualizer(synth_slice)
    ERA5_u_values = era5_slice.ds['u'].values.squeeze()
    ERA5_v_values = era5_slice.ds['v'].values.squeeze()

    ERA5_angles = np.arctan2(ERA5_v_values, ERA5_u_values) * (180 / np.pi)  # Convert to degrees





    uv_colormap = 'seismic'
    diff_colormap = 'inferno'
    angle_colormap = 'hsv'

    latitudes = synth_slice.ds.latitude.values
    longitudes = synth_slice.ds.longitude.values


    # plt.figure(figsize=(20, 15))
    # plt.subplot(1, 3, 1)
    # plt.imshow(SYNTH_u_values, cmap=uv_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
    # plt.colorbar(label="U (m/s)")
    # plt.title(f"U Wind for SYNTH")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.subplot(1, 3, 2)
    # plt.imshow(SYNTH_v_values, cmap=uv_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
    # plt.colorbar(label="U (m/s)")
    # plt.title(f"V Wind for SYNTH")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.subplot(1, 3, 3)
    # plt.imshow(SYNTH_angles, cmap=angle_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()])
    # plt.colorbar(label="U (m/s)")
    # plt.title(f"Angle for SYNTH")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.show()
    # exit()

    # ------------------ PREDICT THE U, V, and direction USING A GAUSSIAN PROCESS REGRESSION --------------

    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)
    X_grid = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

    # observation_steps = [10, 50, 100]
    observation_steps = range(1, 101)
    print(f"Max Observations: {observation_steps[-1]}")
    trial_folder = "70hPa"
    gif_savename = f"animation{trial_folder}"

    # --- COLOR BAR RANGES ---
    diff_speed_min = 0
    diff_speed_max = 20
    diff_direction_min = 0
    diff_direction_max = 60

    u_wind_min = np.min(SYNTH_u_values) - 2
    u_wind_max = np.max(SYNTH_u_values) + 2

    v_wind_min = np.min(SYNTH_v_values) - 2
    v_wind_max = np.max(SYNTH_v_values) + 2

    angle_wind_min = -180
    angle_wind_max = 180

    # Storing mean, std of difference
    u_diff_mean_list = []
    u_diff_std_list = []
    v_diff_mean_list = []
    v_diff_std_list = []
    angle_diff_mean_list = []
    angle_diff_std_list = []

    # Initialize empty lists for observations
    X_sample_list = []
    U_sample_list = []
    V_sample_list = []

    # Function to calculate subplot index
    def subplot_index(row, col, total_cols):
        """
        Calculate the subplot index for plt.subplot() based on row and column.

        Args:
            row (int): Row number (1-based indexing).
            col (int): Column number (1-based indexing).
            total_cols (int): Total number of columns in the grid.

        Returns:
            int: Subplot index for plt.subplot().
        """
        return (row - 1) * total_cols + col

    # Create an evenly spaced grid of ERA5 points
    def create_evenly_spaced_grid(latitudes, longitudes, grid_spacing):
        """
        Create an evenly spaced grid of lat/lon points.

        Args:
            latitudes (np.ndarray): Array of latitude values.
            longitudes (np.ndarray): Array of longitude values.
            grid_spacing (int): Step size for selecting grid points.

        Returns:
            np.ndarray: Grid points as (lat, lon) pairs.
        """
        lat_grid = latitudes[::grid_spacing]
        lon_grid = longitudes[::grid_spacing]
        lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid)
        return np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

    # Initialize the ERA5 grid
    grid_spacing = 3  # Example: use every 5th point in the lat/lon arrays
    X_era5_grid = create_evenly_spaced_grid(latitudes, longitudes, grid_spacing)



    #Here we need to start with the grid initialized to ERA5,  and then add the ranom points
    X_sample_list= X_era5_grid.tolist()
    #samp_x = [lat, lon]  # Create a 1D array for the lat-lon pair
    samp_u_era5 = np.array([
        era5_slice.ds['u'].sel(latitude=lat, longitude=lon, method="nearest").values
        for lat, lon in zip(X_era5_grid[:, 0], X_era5_grid[:, 1])
    ])
    samp_v_era5 = np.array([
        era5_slice.ds['v'].sel(latitude=lat, longitude=lon, method="nearest").values
        for lat, lon in zip(X_era5_grid[:, 0], X_era5_grid[:, 1])
    ])

    U_sample_list = samp_u_era5.tolist()
    V_sample_list = samp_v_era5.tolist()


    for num_samples in observation_steps:
        # Add more observations to the dataset
        print(f"Number of Observations: {num_samples}")
        '''
        for _ in range(len(X_sample_list)-len(X_era5_grid), num_samples):
            lat = np.random.choice(latitudes)  # Random latitude
            lon = np.random.choice(longitudes)  # Random longitude
            samp_x = [lat, lon]  # Create a 1D array for the lat-lon pair
            samp_u = synth_slice.ds['u'].sel(latitude=lat, longitude=lon,
                                             method="nearest").values  # Corresponding u value
            samp_v = synth_slice.ds['v'].sel(latitude=lat, longitude=lon,
                                             method="nearest").values  # Corresponding v value
            X_sample_list.append(samp_x)  # Append lat-lon pair to X_sample
            U_sample_list.append(samp_u[0])  # Append value to Y_sample
            V_sample_list.append(samp_v[0])
        '''

        types_combined = np.array(['noisy'] * len(X_era5_grid)  + ['perfect'] * (len(X_sample_list)-len(X_era5_grid)))  # Label types
        #print(types_combined)

        # Convert lists to numpy arrays
        X_sample = np.array(X_sample_list)  # Shape: (num_samples, 2)
        U_sample = np.array(U_sample_list)  # Shape: (num_samples, 1)
        V_sample = np.array(V_sample_list)  # Shape: (num_samples, 1)

        # Assign noise levels based on type
        noise_levels = np.array([1 ** 2 if t == 'noisy' else .1 ** 2 for t in types_combined])

        # Define GPR models
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        # Equivalent kernel to Gpy
        kernel = C(1.0) * RBF(length_scale=5.0)
        gpr_u = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=noise_levels)
        gpr_v = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=noise_levels)

        # Fit GPR to the initial ERA5 data
        gpr_u.fit(X_sample, U_sample)
        gpr_v.fit(X_sample, V_sample)

        # Generate initial predictions
        U_grid_pred, U_std = gpr_u.predict(X_grid, return_std=True)
        V_grid_pred, V_std = gpr_v.predict(X_grid, return_std=True)
        U_grid_pred = U_grid_pred.reshape(lat_grid.shape)
        V_grid_pred = V_grid_pred.reshape(lat_grid.shape)

        # TRANSPOSE PREDICTION TO MATCH LAT, LON ORDER
        U_grid_pred = U_grid_pred.T
        V_grid_pred = V_grid_pred.T

        # --- COMPUTE PREDICTED ANGLES AND DIFFERENCE

        pred_angles = np.arctan2(V_grid_pred, U_grid_pred) * (180 / np.pi)  # Convert to degrees
        angle_diff = SYNTH_angles - pred_angles
        angle_diff = (angle_diff + 180) % 360 - 180
        angle_diff = np.abs(angle_diff)

        # ------- PLOTTING  ------------

        # --- PLOTTING ERA5 U, V, and angles
        plt.figure(figsize=(20, 10))

        # Define the grid dimensions
        rows, cols = 3, 4


        plt.subplot(rows, cols, subplot_index(1, 1, cols))
        plt.imshow(ERA5_u_values, cmap=uv_colormap, origin='lower',
                   extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=u_wind_min,
                   vmax=u_wind_max)
        plt.colorbar(label="U (m/s)")
        plt.title(f"ERA5 U_WIND (ACTUAL) @:{pres}hPa")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        #plt.scatter(X_sample_lons, X_sample_lats, color='red', marker='x', label='Observations')

        plt.subplot(rows, cols, subplot_index(2, 1, cols))
        plt.imshow(ERA5_v_values, cmap=uv_colormap, origin='lower',
                   extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=v_wind_min,
                   vmax=v_wind_max)
        plt.colorbar(label="V (m/s)")
        plt.title(f"ERA5 V_WIND (ACTUAL) @:{pres}hPa")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        #plt.scatter(X_sample_lons, X_sample_lats, color='red', marker='x', label='Observations')

        plt.subplot(rows, cols, subplot_index(3, 1, cols))
        plt.imshow(ERA5_angles, cmap=angle_colormap, origin='lower',
                   extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=angle_wind_min,
                   vmax=angle_wind_max)
        plt.colorbar(label="Direction (degrees)")
        plt.title(f"ERA5 Wind Direction (ACTUAL) @:{pres}hPa")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        #plt.scatter(X_sample_lons, X_sample_lats, color='red', marker='x', label='Observations')



        # --- PLOTTING ACTUAL U, V, and angles
        plt.subplot(rows, cols, subplot_index(1, 2, cols))
        plt.imshow(SYNTH_u_values, cmap=uv_colormap, origin='lower',
                   extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=u_wind_min,
                   vmax=u_wind_max)
        plt.colorbar(label="U (m/s)")
        plt.title(f"SYNTH U_WIND (ACTUAL) @:{pres}hPa")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')

        plt.subplot(rows, cols, subplot_index(2, 2, cols))
        plt.imshow(SYNTH_v_values, cmap=uv_colormap, origin='lower',
                   extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=v_wind_min,
                   vmax=v_wind_max)
        plt.colorbar(label="V (m/s)")
        plt.title(f"SYNTH V_WIND (ACTUAL) @:{pres}hPa")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')

        plt.subplot(rows, cols, subplot_index(3, 2, cols))
        plt.imshow(SYNTH_angles, cmap=angle_colormap, origin='lower',
                   extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=angle_wind_min,
                   vmax=angle_wind_max)
        plt.colorbar(label="Direction (degrees)")
        plt.title(f"Wind Direction (ACTUAL) @:{pres}hPa")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')



        # --- PLOTTING GPR PREDICTION U, V, and derived angles
        plt.subplot(rows, cols, subplot_index(1, 3, cols))
        img = plt.imshow(U_grid_pred, cmap=uv_colormap, origin='lower',
                         extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=u_wind_min,
                         vmax=u_wind_max)
        cbar = plt.colorbar(img)
        cbar.set_label("Predicted U Wind (m/s)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"GP Predictions of U Wind with {num_samples} Observations")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')
        plt.scatter(X_era5_grid[:, 1], X_era5_grid[:, 0], color='cyan', marker='.', s = 2,  label='Initialization')
        #print(X_grid[:, 1].tolist())
        #print(X_grid[:, 0])
        #print(type([-125, -125, -125]), type(X_grid[:, 1].tolist()))

        plt.subplot(rows, cols, subplot_index(2, 3, cols))
        img = plt.imshow(V_grid_pred, cmap=uv_colormap, origin='lower',
                         extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=v_wind_min,
                         vmax=v_wind_max)
        cbar = plt.colorbar(img)
        cbar.set_label("Predicted V Wind (m/s)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"GP Predictions of V Wind with {num_samples} Observations")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')
        plt.scatter(X_era5_grid[:, 1], X_era5_grid[:, 0], color='cyan', marker='.', s=2, label='Initialization')

        plt.subplot(rows, cols, subplot_index(3, 3, cols))
        img = plt.imshow(pred_angles, cmap=angle_colormap, origin='lower',
                         extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
                         vmin=angle_wind_min, vmax=angle_wind_max)
        cbar = plt.colorbar(img)
        cbar.set_label("(Predicted) Direction (deg)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"GP Prediction of Direcrtion with {num_samples} Observations")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')
        plt.scatter(X_era5_grid[:, 1], X_era5_grid[:, 0], color='cyan', marker='.', s=2, label='Initialization')


        # Validate shape alignment
        assert U_grid_pred.shape == SYNTH_u_values.shape and V_grid_pred.shape == SYNTH_v_values.shape, \
            f"Shapes do not match for U?: {U_grid_pred.shape} vs {SYNTH_u_values.shape}\nShapes do not match for V?: {V_grid_pred.shape} vs {SYNTH_v_values.shape}"

        # Calculate the difference
        u_prediction_diff = np.abs(U_grid_pred - SYNTH_u_values)
        u_mean_diff = np.mean(u_prediction_diff)
        u_std_diff = np.std(u_prediction_diff)

        v_prediction_diff = np.abs(V_grid_pred - SYNTH_v_values)
        v_mean_diff = np.mean(v_prediction_diff)
        v_std_diff = np.std(v_prediction_diff)

        angle_mean_diff = np.mean(angle_diff)
        angle_std_diff = np.std(angle_diff)

        u_diff_mean_list.append(u_mean_diff)
        u_diff_std_list.append(u_std_diff)
        v_diff_mean_list.append(v_mean_diff)
        v_diff_std_list.append(v_std_diff)
        angle_diff_mean_list.append(angle_mean_diff)
        angle_diff_std_list.append(angle_std_diff)

        print(f"With {num_samples} observations:")
        print(f"Mean difference of U: {u_mean_diff}")
        print(f"Standard deviation of difference of U: {u_std_diff}")
        print(f"Mean difference of V: {v_mean_diff}")
        print(f"Standard deviation of difference of V: {v_std_diff}")
        print(f"Mean difference of direction: {angle_mean_diff}")
        print(f"Standard deviation of difference of direction: {angle_std_diff}")

        # --- PLOTTING GPR TO ACTUAL DIFFERENCE OF U AND V and derived difference of angles
        # Plot the difference
        plt.subplot(rows, cols, subplot_index(1, 4, cols))
        img = plt.imshow(u_prediction_diff, cmap=diff_colormap, origin='lower',
                         extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
                         vmin=diff_speed_min, vmax=diff_speed_max)
        cbar = plt.colorbar(img)
        cbar.set_label("Difference of U |Predicted - Actual|")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(
            f"Difference Between U GP Prediction and Actual with {num_samples} Observations\nMean: {u_mean_diff:.3f}, Std: {u_std_diff:.3f}")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')

        plt.subplot(rows, cols, subplot_index(2, 4, cols))
        img = plt.imshow(v_prediction_diff, cmap=diff_colormap, origin='lower',
                         extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
                         vmin=diff_speed_min, vmax=diff_speed_max)
        cbar = plt.colorbar(img)
        cbar.set_label("Difference of V |Predicted - Actual|")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(
            f"Difference Between V GP Prediction and Actual with {num_samples} Observations\nMean: {v_mean_diff:.3f}, Std: {v_std_diff:.3f}")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')

        plt.subplot(rows, cols, subplot_index(3, 4, cols))
        img = plt.imshow(angle_diff, cmap=diff_colormap, origin='lower',
                         extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()],
                         vmin=diff_direction_min, vmax=diff_direction_max)
        cbar = plt.colorbar(img)
        cbar.set_label("Difference of Direction |Predicted - Actual|")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(
            f"Difference Between Direction GP Prediction and Actual with {num_samples} Observations\nMean: {v_mean_diff:.3f}, Std: {v_std_diff:.3f}")
        # Overlay the observation locations
        X_sample_lats = X_sample[:, 0]
        X_sample_lons = X_sample[:, 1]
        plt.scatter(X_sample_lons[len(X_era5_grid):], X_sample_lats[len(X_era5_grid):], color='red', marker='x', label='Observations')

        plt.show()


        plt.tight_layout()
        plt.suptitle(f"{pres} hPA using {num_samples} observations")
        # plt.show()
        plt.savefig(f"temp_images/{trial_folder}/step_{num_samples}.png")
        plt.close()

    images = []
    for i in range(1, observation_steps[-1] + 1):
        images.append(Image.open(f"temp_images/{trial_folder}/step_{i}.png"))

    # Save as GIF
    images[0].save(
        f"temp_images/{trial_folder}/{gif_savename}.gif",
        save_all=True,
        append_images=images[1:],
        duration=500,  # Duration between frames in milliseconds
        loop=0  # Infinite loop
    )

    # Clean up: Remove the temporary images
    # for i in range(observation_steps):
    #     os.remove(f"temp_images/step_{i}.png")

    print(f"GIF saved as '{gif_savename}.gif'")

    u_diff_mean_list = np.array(u_diff_mean_list)
    u_diff_std_list = np.array(u_diff_std_list)
    v_diff_mean_list = np.array(v_diff_mean_list)
    v_diff_std_list = np.array(v_diff_std_list)
    angle_diff_mean_list = np.array(angle_diff_mean_list)
    angle_diff_std_list = np.array(angle_diff_std_list)

    plt.figure(figsize=(10, 10))
    plt.plot(observation_steps, u_diff_mean_list, label="U")
    plt.plot(observation_steps, v_diff_mean_list, label="V")
    plt.plot(observation_steps, angle_diff_mean_list, label="Angle")
    plt.xlabel("Number of Observations")
    plt.ylabel("Mean Difference of Actual and Prediction")
    plt.legend()
    plt.title("Mean Difference as a Function of Observations")
    plt.savefig(f'temp_images/{trial_folder}/mean_diff.png')
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.plot(observation_steps, u_diff_std_list, label="U")
    plt.plot(observation_steps, v_diff_std_list, label="V")
    plt.plot(observation_steps, angle_diff_std_list, label="Angle")
    plt.xlabel("Number of Observations")
    plt.ylabel("Standard Deviation of Difference of Actual and Prediction")
    plt.legend()
    plt.title("Standard Deviation of Difference as a Function of Observations")
    plt.savefig(f'temp_images/{trial_folder}/std_diff.png')
    plt.close()


if __name__ == '__main__':
    gpr_compare()