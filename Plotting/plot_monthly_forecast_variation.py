"""
Plot Monthly Forecast Variation with  mean and std angle differences between Synth and ERA5

This assumes the Synth and ERA5 forecasts were downloaded with the same dimensions
"""

import numpy as np
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast, Forecast_Subset
import matplotlib.pyplot as plt
import pandas as pd
import copy

def monthlyDistribution(synth_filename: str, era5_filename: str, verbose: bool = False, extra_test: bool = False):

    # ****** Function Parameters ******
    # synth_filename: Name of the synthetic wind netcdf file
    # era5_filename: Name of the era5 wind netcdf file
    # verbose: Choose to include details for each 12 hr snapshot of netcdf
    # extra_test: Choose to include time varying plots of Synthetic and ERA5 forecasts individually
    # *********************************

    # **************** LOADING DATA SET *************************
    FORECAST_SYNTH = Forecast(synth_filename,  forecast_type = "SYNTH")


    # Get month associated with Synth
    month =  pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month

    #Then process ERA5 to span the same timespan as a monthly Synthwinds File
    # FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month = month)
    FORECAST_ERA5 = Forecast(era5_filename, forecast_type="ERA5", month = month)
    env_params["rel_dist"] = 10_000_000 #Manually Override relative distance to show a whole subset

    start_time = FORECAST_SYNTH.TIME_MIN
    print(f"Start time: {start_time}")


    forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)
    forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)


    forecast_subset_synth.assign_coord(
        0.5 * (forecast_subset_synth.Forecast.LAT_MAX + forecast_subset_synth.Forecast.LAT_MIN),
        0.5 * (forecast_subset_synth.Forecast.LON_MAX + forecast_subset_synth.Forecast.LON_MIN),
        start_time)
    forecast_subset_synth.subset_forecast(days=31)


    forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)
    forecast_subset_era5.assign_coord(forecast_subset_synth.lat_central, forecast_subset_synth.lon_central, start_time)
    # forecast_subset.randomize_coord()
    forecast_subset_era5.subset_forecast(days=31)

    print("TEST *****************")
    print("ERA5")
    print(forecast_subset_era5.ds.time.values)
    print("SYNTH")
    print(forecast_subset_synth.ds.time.values)

    time_list = forecast_subset_synth.ds.time.values

    mean_arr = []
    std_arr = []
    if extra_test:
        #Extra test
        era5_mean_arr = []
        era5_std_arr = []
        synth_mean_arr =[]
        synth_std_arr = []
        # end extra test

    for time in time_list:

        mean_by_level = []
        std_by_level = []

        if extra_test:
            # Extra test
            era5_std = []
            era5_mean = []
            synth_std = []
            synth_mean = []
            # end extra test
        
        print(f"Time: {time}")

        for i in range(1, forecast_subset_era5.level_dim-1):

            pres = forecast_subset_era5.ds.level.values[i]
            if verbose:
                print(pres)
            alt_era5 = forecast_subset_era5.get_alt_from_pressure(pres)
            # For Synthwinds, can Assume every coordinate has the same altitude column, so just take first index
            alt_column = forecast_subset_synth.ds.sel(time=time).isel(latitude=0, longitude=0)['z'].values / 9.81
            # Find the index of the nearest z value
            nearest_idx = np.argmin(np.abs(alt_column - alt_era5))
            if verbose:
                print(pres, alt_era5, forecast_subset_synth.ds.sel(time=time).isel(latitude=0, longitude=0)['z'].values / 9.81)


            #Take some slices of the forecasts for plotting
            era_5_slice = copy.deepcopy(forecast_subset_era5)
            era_5_slice.ds =  forecast_subset_era5.ds.isel(level=slice(i, i+1))
            era_5_slice.ds =  era_5_slice.ds.sel(time=time)
            if verbose:
                print(f"ERA5 Slice at {era_5_slice.ds.level.values} -> {era_5_slice.ds['z'].values}")
                print(era_5_slice.ds)
            #forecast_visualizer_era5 = ForecastVisualizer(era_5_slice)
            ERA5_u_values = era_5_slice.ds['u'].values.squeeze()
            ERA5_v_values = era_5_slice.ds['v'].values.squeeze()


            synth_slice = copy.deepcopy(forecast_subset_synth)
            synth_slice.ds = forecast_subset_synth.ds.isel(level=slice(nearest_idx, nearest_idx+1))
            synth_slice.ds =  synth_slice.ds.sel(time=time)
            #forecast_visualizer_synth = ForecastVisualizer(synth_slice)
            SYNTH_u_values = synth_slice.ds['u'].values.squeeze()
            SYNTH_v_values = synth_slice.ds['v'].values.squeeze()

            ERA5_angles = np.arctan2(ERA5_v_values, ERA5_u_values) * (180 / np.pi)  # Convert to degrees
            SYNTH_angles = np.arctan2(SYNTH_v_values, SYNTH_u_values) * (180 / np.pi)  # Convert to degrees


            
            angle_diff = SYNTH_angles - ERA5_angles


            angle_diff = (angle_diff + 180) % 360 - 180
            angle_diff = np.abs(angle_diff)

            mean_by_level = np.append(mean_by_level, np.mean(angle_diff))
            std_by_level = np.append(std_by_level, np.std(angle_diff))

            if extra_test:
                # Extra test
                era5_mean = np.append(era5_mean, np.mean(ERA5_angles))
                era5_std = np.append(era5_std, np.std(ERA5_angles))
                synth_mean = np.append(synth_mean, np.mean(SYNTH_angles))
                synth_std = np.append(synth_std, np.std(SYNTH_angles))
                # end extra test


        mean_arr.append(mean_by_level)
        std_arr.append(std_by_level)

        if extra_test:
            # Extra test
            era5_mean_arr.append(era5_mean)
            era5_std_arr.append(era5_std)
            synth_mean_arr.append(synth_mean)
            synth_std_arr.append(synth_std)
            # end extra test

    mean_arr = np.array(mean_arr)
    std_arr = np.array(std_arr)

    if extra_test:
        # Extra test
        era5_mean_arr = np.array(era5_mean_arr)
        era5_std_arr = np.array(era5_std_arr)
        synth_mean_arr = np.array(synth_mean_arr)
        synth_std_arr = np.array(synth_std_arr)
        # end extra test

    print("Mean Array by Level and Time:")
    print(mean_arr)
    print("Std Dev Array by Level and Time:")
    print(std_arr)

    time_values = pd.to_datetime(time_list)

    # Transpose mean_arr to get the means for each pressure level across time
    mean_arr_T = mean_arr.T  # Now rows represent pressure levels, columns represent times
    std_arr_T = std_arr.T

    # Get the pressure levels from the ERA5 dataset
    pressure_levels = forecast_subset_era5.ds.level.values[1:-1]  # Assuming you skipped first and last levels in the loop

    # Plotting
    plt.figure(figsize=(20, 5))

    for i, pressure in enumerate(pressure_levels):
        plt.plot(time_values, mean_arr_T[i], label=f"{pressure} hPa")  # Plot each pressure level's mean values
        plt.fill_between(time_values, mean_arr_T[i] - std_arr_T[i], mean_arr_T[i] + std_arr_T[i], alpha=0.2)

    # Add labels and legend
    plt.xlabel('Datetime', fontsize=18)
    plt.ylabel('Angle Difference (Degrees)', fontsize=18)

    #plt.title(f'Mean Angle Difference Over Time by Pressure Level {pd.to_datetime(time_list[0]).month_name()}')
    plt.legend(title="Pressure Level (hPa)", loc='upper right')

    plt.xticks(rotation=45, fontsize=16)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=16)
    plt.xlim([time_values[0], time_values[-1]])
    plt.ylim([0, 130])
    plt.tight_layout()

    # Show the plot
    #plt.show()
    #plt.close()

    plt.figure(figsize=(12, 6))

    for i, pressure in enumerate(pressure_levels):
        plt.plot(time_values, std_arr_T[i], label=f"{pressure} hPa")  # Plot each pressure level's mean values

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Standard Deviation Angle Difference (Degrees)')
    plt.title(f'Standard Deviation of Angle Difference Over Time by Pressure Level {pd.to_datetime(time_list[0]).month_name()}')
    plt.legend(title="Pressure Level (hPa)", loc='upper right')

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Show the plot
    #plt.show()


    if extra_test:

        # EXTRA TEST
        era5_mean_arr_T = era5_mean_arr.T  # Now rows represent pressure levels, columns represent times
        era5_std_arr_T = era5_std_arr.T

        # Get the pressure levels from the ERA5 dataset
        pressure_levels = forecast_subset_era5.ds.level.values[1:-1]  # Assuming you skipped first and last levels in the loop

        # Plotting
        plt.figure(figsize=(12, 6))

        for i, pressure in enumerate(pressure_levels):
            plt.plot(time_values, era5_mean_arr_T[i], label=f"{pressure} hPa")  # Plot each pressure level's mean values

        # Add labels and legend
        plt.xlabel('Time')
        plt.ylabel('Mean Angle (Degrees)')
        plt.title(f'Mean Angle ERA5 Over Time by Pressure Level {pd.to_datetime(time_list[0]).month_name()}')
        plt.legend(title="Pressure Level (hPa)", loc='upper right')

        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Show the plot
        #plt.show()
        #plt.close()

        plt.figure(figsize=(12, 6))

        for i, pressure in enumerate(pressure_levels):
            plt.plot(time_values, era5_std_arr_T[i], label=f"{pressure} hPa")  # Plot each pressure level's mean values

        # Add labels and legend
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation Angle (Degrees)')
        plt.title(f'Standard Deviation of Angle ERA5 Over Time by Pressure Level {pd.to_datetime(time_list[0]).month_name()}')
        plt.legend(title="Pressure Level (hPa)", loc='upper right')

        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Show the plot
        #plt.show()

        synth_mean_arr_T = synth_mean_arr.T  # Now rows represent pressure levels, columns represent times
        synth_std_arr_T = synth_std_arr.T

        # Get the pressure levels from the ERA5 dataset
        pressure_levels = forecast_subset_era5.ds.level.values[1:-1]  # Assuming you skipped first and last levels in the loop

        # Plotting
        plt.figure(figsize=(12, 6))

        for i, pressure in enumerate(pressure_levels):
            plt.plot(time_values, synth_mean_arr_T[i], label=f"{pressure} hPa")  # Plot each pressure level's mean values

        # Add labels and legend
        plt.xlabel('Time')
        plt.ylabel('Mean Angle SYNTH (Degrees)')
        plt.title(f'Mean Angle SYNTH Over Time by Pressure Level {pd.to_datetime(time_list[0]).month_name()}')
        plt.legend(title="Pressure Level (hPa)", loc='upper right')

        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Show the plot
        #plt.show()
        #plt.close()

        plt.figure(figsize=(12, 6))

        for i, pressure in enumerate(pressure_levels):
            plt.plot(time_values, synth_std_arr_T[i], label=f"{pressure} hPa")  # Plot each pressure level's mean values

        # Add labels and legend
        plt.xlabel('Time')
        plt.ylabel('Standard Deviation Angle SYNTH (Degrees)')
        plt.title(f'Standard Deviation of Angle SYNTH Over Time by Pressure Level {pd.to_datetime(time_list[0]).month_name()}')
        plt.legend(title="Pressure Level (hPa)", loc='upper right')

        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Show the plot
        #plt.show()

        # END EXTRA TEST


if __name__ == '__main__':
    monthlyDistribution(synth_filename = "SYNTH-Jul-2023-USA-UPDATED.nc", era5_filename="ERA5-H2-2023-USA.nc" ,verbose=False, extra_test=False)
    plt.show()


