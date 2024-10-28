"""
Plots directional wind variances with colored contour plots for each pressure level, as well as the angle differences
between ERA5 and Synth Forecasts.

This assumes the Synth and ERA5 forecasts were downloaded with the same dimensions

"""

import numpy as np
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast, Forecast_Subset
import matplotlib.pyplot as plt
import pandas as pd
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compare(synth_filename: str, era5_filename: str, timestamp: str, individual_plots: bool, speed_plots: bool):

    # ****** Function Parameters ******
    # synth_filename: Name of the synthetic wind netcdf file
    # era5_filename: Name of the era5 wind netcdf file
    # timestamp: Select time at which to compare forecasts
    # individual_plots: Choose to plot comparison individually (for downloading plots)
    # speed_plots: Choose to include velocity comparison
    # *********************************

    print(env_params['era_netcdf'])

    FORECAST_SYNTH = Forecast(synth_filename,  forecast_type = "SYNTH")

    # Get month associated with Synth
    month =  pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month

    #Then process ERA5 to span the same timespan as a monthly Synthwinds File
    FORECAST_ERA5 = Forecast(era5_filename, forecast_type="ERA5", month = month)

    env_params["rel_dist"] = 10_000_000 #Manually Override relative distance to show a whole subset

    timestamp = timestamp

    forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)
    forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)

    forecast_subset_synth.assign_coord(
        0.5 * (forecast_subset_synth.Forecast.LAT_MAX + forecast_subset_synth.Forecast.LAT_MIN),
        0.5 * (forecast_subset_synth.Forecast.LON_MAX + forecast_subset_synth.Forecast.LON_MIN),
        timestamp)
    forecast_subset_synth.subset_forecast(days=1)


    forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)
    forecast_subset_era5.assign_coord(forecast_subset_synth.lat_central, forecast_subset_synth.lon_central, timestamp)
    forecast_subset_era5.subset_forecast(days=1)


    # FIND ALTITUDE FOR COMPARISON WITH SYNTH WINDS EXAMPLE USAGE
    """Find nearest pressure levels"""
    alt_era5 = forecast_subset_era5.get_alt_from_pressure(30)

    print("alt_era5", alt_era5)

    # For Synthwinds, can Assume every coordinate has the same altitude column, so just take first index
    alt_column = forecast_subset_synth.ds.isel(time=0, latitude=0, longitude=0)['z'].values/9.81

    print("synth alts", alt_column)
    print("Num ERA5 LEVELS: ", forecast_subset_era5.ds.level.values)
    print("Num SYNTH LEVELS: ", len(alt_column))

    # Find the index of the nearest z value
    nearest_idx = np.argmin(np.abs(alt_column - alt_era5))

    print("synth_idx", nearest_idx,forecast_subset_synth.ds.isel(time=0, latitude=0, level=nearest_idx, longitude=0)['z'].values/9.81)
    print("*****DONE ****")

    skip = 2

    #Now let's make a side by side GIF
    #Leaving off 20 hpa and 150 hpa since synthwinds doesn't include those
    for i in range(1, forecast_subset_era5.level_dim-1):

        pres = forecast_subset_era5.ds.level.values[i]
        print(pres)
        alt_era5 = forecast_subset_era5.get_alt_from_pressure(pres)
        # For Synthwinds, can Assume every coordinate has the same altitude column, so just take first index
        alt_column = forecast_subset_synth.ds.sel(time=timestamp).isel(latitude=0, longitude=0)['z'].values / 9.81
        # Find the index of the nearest z value
        nearest_idx = np.argmin(np.abs(alt_column - alt_era5))
        print(pres, alt_era5, forecast_subset_synth.ds.sel(time=timestamp).isel(latitude=0, longitude=0)['z'].values / 9.81)


        #Take some slices of the forecasts for plotting
        era_5_slice = copy.deepcopy(forecast_subset_era5)
        era_5_slice.ds =  forecast_subset_era5.ds.isel(level=slice(i, i+1))
        era_5_slice.ds =  era_5_slice.ds.sel(time=timestamp)
        print(f"ERA5 Slice at {era_5_slice.ds.level.values} -> {era_5_slice.ds['z'].values}")
        print(era_5_slice.ds)
        #forecast_visualizer_era5 = ForecastVisualizer(era_5_slice)
        ERA5_u_values = era_5_slice.ds['u'].values.squeeze()
        ERA5_v_values = era_5_slice.ds['v'].values.squeeze()


        synth_slice = copy.deepcopy(forecast_subset_synth)
        synth_slice.ds = forecast_subset_synth.ds.isel(level=slice(nearest_idx, nearest_idx+1))
        synth_slice.ds =  synth_slice.ds.sel(time=timestamp)
        #forecast_visualizer_synth = ForecastVisualizer(synth_slice)
        SYNTH_u_values = synth_slice.ds['u'].values.squeeze()
        SYNTH_v_values = synth_slice.ds['v'].values.squeeze()

        ERA5_angles = np.arctan2(ERA5_v_values, ERA5_u_values) * (180 / np.pi)  # Convert to degrees
        SYNTH_angles = np.arctan2(SYNTH_v_values, SYNTH_u_values) * (180 / np.pi)  # Convert to degrees
        #SYNTH_angles = ERA5_angles + 50

        # Calculate angle difference
        angle_diff = SYNTH_angles - ERA5_angles

        # Normalize the angle difference to be within [-180, 180] degrees
        angle_diff = (angle_diff + 180) % 360 - 180
        angle_diff = np.abs(angle_diff)

        latitudes = era_5_slice.ds.latitude.values
        longitudes = era_5_slice.ds.longitude.values

        print(era_5_slice.ds)

        max_uv_val = np.max([np.max(np.abs(ERA5_u_values)), np.max(np.abs(SYNTH_u_values)), np.max(np.abs(ERA5_v_values)), np.max(np.abs(SYNTH_v_values))])
        #min_u_val = np.min([np.min(ERA5_u_values), np.min(SYNTH_u_values)])
        max_diff_val = np.max([np.max(abs(SYNTH_u_values-ERA5_u_values)), np.max(abs(SYNTH_v_values-ERA5_v_values))])
        max_angle_diff_val = np.max(angle_diff)

        uv_colormap = 'seismic'
        diff_colormap = 'inferno'

        if speed_plots:

            plt.figure(figsize=(20, 10))
            plt.subplot(2, 3, 1)
            plt.imshow(ERA5_u_values, cmap=uv_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-max_uv_val, vmax=max_uv_val)
            plt.colorbar(label="U (m/s)")
            plt.title(f"U Wind for ERA5")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.subplot(2, 3, 2)
            plt.imshow(SYNTH_u_values, cmap=uv_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-max_uv_val, vmax=max_uv_val)
            plt.colorbar(label="U (m/s)")
            plt.title(f"U Wind for SYNTH")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.subplot(2, 3, 3)
            plt.imshow(abs(SYNTH_u_values-ERA5_u_values), cmap=diff_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=0, vmax=max_diff_val)
            plt.colorbar(label="U (m/s)")
            plt.title(f"U Difference Between SYNTH and ERA5")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

            plt.subplot(2, 3, 4)
            plt.imshow(ERA5_v_values, cmap=uv_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-max_uv_val, vmax=max_uv_val)
            plt.colorbar(label="V (m/s)")
            plt.title(f"V Wind for ERA5")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.subplot(2, 3, 5)
            plt.imshow(SYNTH_v_values, cmap=uv_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-max_uv_val, vmax=max_uv_val)
            plt.colorbar(label="V (m/s)")
            plt.title(f"V Wind for SYNTH")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.subplot(2, 3, 6)
            plt.imshow(abs(SYNTH_v_values-ERA5_v_values), cmap=diff_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=0, vmax=max_diff_val)
            plt.colorbar(label="V (m/s)")
            plt.title(f"V Difference Between SYNTH and ERA5")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

            plt.tight_layout()
            plt.suptitle(f"{timestamp} at {era_5_slice.ds.level.values} hPa")

            # Display the heatmap
            plt.show()
            plt.close()

            plt.figure(figsize=(20, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(abs(SYNTH_u_values-ERA5_u_values), cmap=diff_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=0, vmax=max_diff_val)
            plt.colorbar(label="U (m/s)")
            plt.title(f"U Difference Between SYNTH and ERA5")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.subplot(1, 3, 2)
            plt.imshow(abs(SYNTH_v_values-ERA5_v_values), cmap=diff_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=0, vmax=max_diff_val)
            plt.colorbar(label="V (m/s)")
            plt.title(f"V Difference Between SYNTH and ERA5")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.subplot(1, 3, 3)
            plt.imshow(angle_diff, cmap=diff_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=0, vmax=max_angle_diff_val)
            plt.colorbar(label="Angle (Degrees)")
            plt.title(f"Angle Difference between SYNTH and ERA5")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

            plt.tight_layout()
            plt.suptitle(f"{timestamp} at {era_5_slice.ds.level.values} hPa")
            plt.show()
            plt.close()

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(ERA5_angles, cmap='hsv', origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-180, vmax=180)
        bar1 = plt.colorbar(shrink =  0.6)
        bar1.set_label("Direction (Degrees)", fontsize=12)
        #plt.title(f"ERA5 Direction")
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.subplot(1, 3, 2)
        plt.imshow(SYNTH_angles, cmap='hsv', origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-180, vmax=180)
        bar2 = plt.colorbar(shrink =  0.6)
        bar2.set_label("Direction (Degrees)", fontsize=12)
        #plt.title(f"SYNTH Direction")
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.subplot(1, 3, 3)
        plt.imshow(angle_diff, cmap=diff_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=0, vmax=180)
        bar3 = plt.colorbar(shrink =  0.6)
        bar3.set_label("Direction (Degrees)", fontsize=12)
        #plt.title(f"Angle Difference (Absolute)")
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)

        plt.tight_layout()
        #plt.suptitle(f"{timestamp} at {era_5_slice.ds.level.values} hPa")
        plt.show()
        plt.close()


        if individual_plots:
            plt.figure(figsize=(8, 5))
            ax1 = plt.gca()
            im1 = ax1.imshow(ERA5_angles, cmap='hsv', origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-180, vmax=180)

            plt.xlabel("Longitude", fontsize=20)
            plt.ylabel("Latitude", fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Use make_axes_locatable to create a colorbar matching the height of the plot
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.2)  # Adjust size and padding as necessary
            bar1 = plt.colorbar(im1, cax=cax1)
            bar1.set_label("Direction (Degrees)", fontsize=20)
            bar1.ax.tick_params(labelsize=16)

            tick_interval = 60  # Set the interval between ticks (e.g., every 60 degrees)
            bar1.set_ticks(np.arange(-180, 181, tick_interval))


            plt.tight_layout()
            plt.show()
            plt.close()

            # Plot SYNTH Angles
            plt.figure(figsize=(8, 5))
            ax2 = plt.gca()
            im2 = ax2.imshow(SYNTH_angles, cmap='hsv', origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=-180, vmax=180)
            plt.xlabel("Longitude", fontsize=20)
            plt.ylabel("Latitude", fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Create a matching colorbar for SYNTH Angles
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.2)
            bar2 = plt.colorbar(im2, cax=cax2)
            bar2.set_label("Direction (Degrees)", fontsize=20)
            bar2.ax.tick_params(labelsize=16)

            tick_interval = 60  # Set the interval between ticks (e.g., every 60 degrees)
            bar2.set_ticks(np.arange(-180, 181, tick_interval))
            plt.tight_layout()
            plt.show()
            plt.close()

            # Plot Angle Difference
            plt.figure(figsize=(8, 5))
            ax3 = plt.gca()
            im3 = ax3.imshow(angle_diff, cmap=diff_colormap, origin='lower', extent=[longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], vmin=0, vmax=180)
            plt.xlabel("Longitude", fontsize=20)
            plt.ylabel("Latitude", fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Create a matching colorbar for Angle Difference
            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes("right", size="5%", pad=0.2)
            bar3 = plt.colorbar(im3, cax=cax3)
            bar3.set_label("Direction (Degrees)", fontsize=20)
            bar3.ax.tick_params(labelsize=16)

            tick_interval = 30  # Set the interval between ticks (e.g., every 60 degrees)
            bar3.set_ticks(np.arange(0, 181, tick_interval))

            plt.tight_layout()
            plt.show()
            plt.close()

        print(f"Mean Difference: {np.mean(angle_diff)}")
        print(f"Std Dev Difference: {np.std(angle_diff)}")

if __name__ == '__main__':
    compare(synth_filename="SYNTH-Jul-2023-USA-UPDATED.nc", era5_filename="ERA5-H2-2023-USA.nc", timestamp="2023-07-17T00:00:00.000000000", individual_plots=False, speed_plots = True)
