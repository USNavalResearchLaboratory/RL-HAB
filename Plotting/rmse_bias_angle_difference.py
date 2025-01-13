import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
from pandas.tseries.offsets import MonthEnd
import math
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast, Forecast_Subset
import csv


def extract_radiosonde_locations(filename):
    def convert_lat_lon(lat, lat_dir, lon, lon_dir):
        new_lat = -1 * lat if lat_dir == 'S' else 1 * lat
        new_lon = -1 * lon if lon_dir == 'W' else 1 * lon
        return new_lat, new_lon

    lat_lon_list = []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            lat = float(row['LAT'])
            lon = float(row['LONG'])
            lat_dir = str(row['N'])
            lon_dir = str(row['E'])
            new_lat, new_lon = convert_lat_lon(lat, lat_dir, lon, lon_dir)
            lat_lon_list.append((new_lat, new_lon))

    return lat_lon_list


def compute_latitude_weights(latitudes):
    """Compute normalized latitude weights given a 1D array of latitudes."""
    weights = np.cos(np.deg2rad(latitudes))
    return weights / weights.mean()


def compute_weighted_rmse(fc_values, true_values, latitudes):
    """
    Compute latitude-weighted RMSE.
    """
    if fc_values.ndim == 1:
        # Radiosonde mode
        error = fc_values - true_values
        weights = compute_latitude_weights(latitudes)
        weighted_squared_error = (error ** 2) * weights
        return np.sqrt(np.mean(weighted_squared_error))
    else:
        # Region mode (2D arrays)
        error = fc_values - true_values
        weights = compute_latitude_weights(latitudes)[:, np.newaxis]
        weighted_squared_error = (error ** 2) * weights
        return np.sqrt(np.mean(weighted_squared_error))


def compute_metrics_for_period(
        synth_filenames, era5_filenames, start_times, end_times, radiosonde_locations=None, region_mode=False
):
    """
    Compute weighted RMSE, bias, angle difference, and magnitude difference (absolute) for either radiosonde locations or entire region.
    """
    monthly_rmse_u_list = []
    monthly_rmse_v_list = []
    monthly_bias_u_list = []
    monthly_bias_v_list = []
    monthly_angle_diff_list = []
    monthly_magnitude_diff_list = []
    monthly_timestamps_list = []

    n_months = len(synth_filenames)

    # Determine ERA5 file for the first month
    first_synth_file = synth_filenames[0]
    first_month = start_times[0].month
    if n_months == 12:
        era5_file = era5_filenames[0] if first_month <= 6 else era5_filenames[1]
    else:
        era5_file = era5_filenames[0]

    FORECAST_SYNTH_TEMPLATE = Forecast(first_synth_file, forecast_type="SYNTH")
    FORECAST_ERA5_TEMPLATE = Forecast(era5_file, forecast_type="ERA5")

    env_params["rel_dist"] = 10_000_000

    forecast_subset_synth_template = Forecast_Subset(FORECAST_SYNTH_TEMPLATE)
    forecast_subset_era5_template = Forecast_Subset(FORECAST_ERA5_TEMPLATE)

    timestamp_initial_template = pd.to_datetime(FORECAST_SYNTH_TEMPLATE.TIME_MIN)
    forecast_subset_synth_template.assign_coord(
        0.5 * (forecast_subset_synth_template.Forecast.LAT_MAX + forecast_subset_synth_template.Forecast.LAT_MIN),
        0.5 * (forecast_subset_synth_template.Forecast.LON_MAX + forecast_subset_synth_template.Forecast.LON_MIN),
        timestamp_initial_template)
    forecast_subset_synth_template.subset_forecast(days=1)

    forecast_subset_era5_template.assign_coord(
        forecast_subset_synth_template.lat_central,
        forecast_subset_synth_template.lon_central,
        timestamp_initial_template)
    forecast_subset_era5_template.subset_forecast(days=1)

    print("forecast_subset", forecast_subset_era5_template.ds)

    pressure_levels = forecast_subset_era5_template.ds.level.values[1:-1]
    num_levels = len(pressure_levels)

    print("pressure levels", pressure_levels)

    lat_min = float(forecast_subset_era5_template.ds.latitude.min())
    lat_max = float(forecast_subset_era5_template.ds.latitude.max())
    lon_min = float(forecast_subset_era5_template.ds.longitude.min())
    lon_max = float(forecast_subset_era5_template.ds.longitude.max())

    print("Where stuck?")
    print("region_mode", region_mode)

    if not region_mode:
        # Radiosonde mode
        valid_locations = [(lat, lon) for lat, lon in radiosonde_locations
                           if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max]

    for month_idx in range(n_months):
        synth_file = synth_filenames[month_idx]
        if synth_file is None:  # Handle missing files
            # Fill with NaN for the missing period
            timestamps = pd.date_range(start=start_times[month_idx], end=end_times[month_idx], freq="12H")
            nan_matrix = np.full((num_levels, len(timestamps)), np.nan)

            monthly_rmse_u_list.append(nan_matrix)
            monthly_rmse_v_list.append(nan_matrix)
            monthly_bias_u_list.append(nan_matrix)
            monthly_bias_v_list.append(nan_matrix)
            monthly_angle_diff_list.append(nan_matrix)
            monthly_magnitude_diff_list.append(nan_matrix)
            monthly_timestamps_list.append(timestamps)
            continue

        if len(era5_filenames)>1:
            era5_file = era5_filenames[0] if start_times[month_idx].month <= 6 else era5_filenames[1]
        else:
            era5_file = era5_filenames[0]
        FORECAST_SYNTH = Forecast(synth_file, forecast_type="SYNTH")
        FORECAST_ERA5 = Forecast(era5_file, forecast_type="ERA5")

        forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)
        forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)

        timestamp_initial = pd.to_datetime(FORECAST_SYNTH.TIME_MIN)
        forecast_subset_synth.assign_coord(
            0.5 * (forecast_subset_synth.Forecast.LAT_MAX + forecast_subset_synth.Forecast.LAT_MIN),
            0.5 * (forecast_subset_synth.Forecast.LON_MAX + forecast_subset_synth.Forecast.LON_MIN),
            timestamp_initial)
        forecast_subset_synth.subset_forecast(days=1)

        forecast_subset_era5.assign_coord(
            forecast_subset_synth.lat_central,
            forecast_subset_synth.lon_central,
            timestamp_initial)
        forecast_subset_era5.subset_forecast(days=1)

        timestamps = pd.date_range(start=start_times[month_idx], end=end_times[month_idx], freq="12h")

        rmse_u_matrix_month = np.full((num_levels, len(timestamps)), np.nan)
        rmse_v_matrix_month = np.full((num_levels, len(timestamps)), np.nan)
        bias_u_matrix_month = np.full((num_levels, len(timestamps)), np.nan)
        bias_v_matrix_month = np.full((num_levels, len(timestamps)), np.nan)
        angle_diff_matrix_month = np.full((num_levels, len(timestamps)), np.nan)
        magnitude_diff_matrix_month = np.full((num_levels, len(timestamps)), np.nan)

        for t_idx, timestamp in enumerate(timestamps):
            print(t_idx, timestamp)
            try:
                forecast_subset_synth.assign_coord(
                    0.5 * (forecast_subset_synth.Forecast.LAT_MAX + forecast_subset_synth.Forecast.LAT_MIN),
                    0.5 * (forecast_subset_synth.Forecast.LON_MAX + forecast_subset_synth.Forecast.LON_MIN),
                    timestamp)
                forecast_subset_synth.subset_forecast(days=1)

                forecast_subset_era5.assign_coord(
                    forecast_subset_synth.lat_central,
                    forecast_subset_synth.lon_central,
                    timestamp)
                forecast_subset_era5.subset_forecast(days=1)

                for i, pres in enumerate(pressure_levels):
                    alt_era5 = forecast_subset_era5.get_alt_from_pressure(pres)
                    alt_column = forecast_subset_synth.ds.sel(time=timestamp, method='nearest') \
                                     .isel(latitude=0, longitude=0)['z'].values / 9.81
                    nearest_idx = np.argmin(np.abs(alt_column - alt_era5))

                    era_5_slice = copy.deepcopy(forecast_subset_era5)
                    era_5_slice.ds = forecast_subset_era5.ds.isel(level=slice(i, i + 1)).sel(time=timestamp)
                    ERA5_u = era_5_slice.ds['u'].values.squeeze()
                    ERA5_v = era_5_slice.ds['v'].values.squeeze()

                    synth_slice = copy.deepcopy(forecast_subset_synth)
                    synth_slice.ds = forecast_subset_synth.ds.isel(level=slice(nearest_idx, nearest_idx + 1)) \
                        .sel(time=timestamp, method='nearest')
                    SYNTH_u = synth_slice.ds['u'].values.squeeze()
                    SYNTH_v = synth_slice.ds['v'].values.squeeze()

                    ERA5_magnitude = np.sqrt(ERA5_u ** 2 + ERA5_v ** 2)
                    SYNTH_magnitude = np.sqrt(SYNTH_u ** 2 + SYNTH_v ** 2)

                    if region_mode:
                        latitudes = era_5_slice.ds.latitude.values
                        rmse_u_val = compute_weighted_rmse(SYNTH_u, ERA5_u, latitudes)
                        rmse_v_val = compute_weighted_rmse(SYNTH_v, ERA5_v, latitudes)

                        bias_u_val = np.mean(SYNTH_u - ERA5_u)
                        bias_v_val = np.mean(SYNTH_v - ERA5_v)

                        ERA5_angles = np.arctan2(ERA5_v, ERA5_u) * (180 / np.pi)
                        SYNTH_angles = np.arctan2(SYNTH_v, SYNTH_u) * (180 / np.pi)
                        angle_diff = SYNTH_angles - ERA5_angles
                        angle_diff = (angle_diff + 180) % 360 - 180
                        angle_diff = np.abs(angle_diff)
                        angle_diff_val = np.mean(angle_diff)

                        # Now take absolute value of magnitude difference
                        magnitude_diff_val = np.mean(np.abs(SYNTH_magnitude - ERA5_magnitude))

                    else:
                        # Radiosonde mode
                        lat_vals = []
                        u_truth_vals = []
                        u_fc_vals = []
                        v_truth_vals = []
                        v_fc_vals = []
                        angle_diffs = []
                        mag_diffs = []

                        for (rlat, rlon) in valid_locations:
                            era_point = era_5_slice.ds.sel(latitude=rlat, longitude=rlon, method='nearest')
                            e_u = era_point['u'].values.squeeze()
                            e_v = era_point['v'].values.squeeze()

                            synth_point = synth_slice.ds.sel(latitude=rlat, longitude=rlon, method='nearest')
                            s_u = synth_point['u'].values.squeeze()
                            s_v = synth_point['v'].values.squeeze()

                            u_truth_vals.append(e_u)
                            v_truth_vals.append(e_v)
                            u_fc_vals.append(s_u)
                            v_fc_vals.append(s_v)
                            lat_vals.append(rlat)

                            ERA5_angle = np.arctan2(e_v, e_u) * (180 / np.pi)
                            SYNTH_angle = np.arctan2(s_v, s_u) * (180 / np.pi)
                            adiff = SYNTH_angle - ERA5_angle
                            adiff = (adiff + 180) % 360 - 180
                            adiff = np.abs(adiff)
                            angle_diffs.append(adiff)

                            ERA5_mag = np.sqrt(e_u ** 2 + e_v ** 2)
                            SYNTH_mag = np.sqrt(s_u ** 2 + s_v ** 2)
                            # Absolute value of the difference
                            mag_diffs.append(np.abs(SYNTH_mag - ERA5_mag))

                        if len(u_truth_vals) == 0:
                            continue

                        u_truth_vals = np.array(u_truth_vals)
                        v_truth_vals = np.array(v_truth_vals)
                        u_fc_vals = np.array(u_fc_vals)
                        v_fc_vals = np.array(v_fc_vals)
                        lat_vals = np.array(lat_vals)
                        angle_diffs = np.array(angle_diffs)
                        mag_diffs = np.array(mag_diffs)

                        rmse_u_val = compute_weighted_rmse(u_fc_vals, u_truth_vals, lat_vals)
                        rmse_v_val = compute_weighted_rmse(v_fc_vals, v_truth_vals, lat_vals)

                        bias_u_val = np.mean(u_fc_vals - u_truth_vals)
                        bias_v_val = np.mean(v_fc_vals - v_truth_vals)
                        angle_diff_val = np.mean(angle_diffs)
                        magnitude_diff_val = np.mean(mag_diffs)

                    rmse_u_matrix_month[i, t_idx] = rmse_u_val
                    rmse_v_matrix_month[i, t_idx] = rmse_v_val
                    bias_u_matrix_month[i, t_idx] = bias_u_val
                    bias_v_matrix_month[i, t_idx] = bias_v_val
                    angle_diff_matrix_month[i, t_idx] = angle_diff_val
                    magnitude_diff_matrix_month[i, t_idx] = magnitude_diff_val

            except KeyError:
                print("pass")
                continue

        monthly_rmse_u_list.append(rmse_u_matrix_month)
        monthly_rmse_v_list.append(rmse_v_matrix_month)
        monthly_bias_u_list.append(bias_u_matrix_month)
        monthly_bias_v_list.append(bias_v_matrix_month)
        monthly_angle_diff_list.append(angle_diff_matrix_month)
        monthly_magnitude_diff_list.append(magnitude_diff_matrix_month)
        monthly_timestamps_list.append(timestamps)

    rmse_u_matrix = np.concatenate(monthly_rmse_u_list, axis=1)
    rmse_v_matrix = np.concatenate(monthly_rmse_v_list, axis=1)
    bias_u_matrix = np.concatenate(monthly_bias_u_list, axis=1)
    bias_v_matrix = np.concatenate(monthly_bias_v_list, axis=1)
    angle_diff_matrix = np.concatenate(monthly_angle_diff_list, axis=1)
    magnitude_diff_matrix = np.concatenate(monthly_magnitude_diff_list, axis=1)
    all_timestamps = pd.concat([pd.Series(t) for t in monthly_timestamps_list]).values

    return all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix, angle_diff_matrix, magnitude_diff_matrix


def plot_results(all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix,
                 angle_diff_matrix, magnitude_diff_matrix, region_mode):
    if region_mode:
        title_region = "Region"
    else:
        title_region = "Radiosonde"

    # Compute min/max for RMSE
    valid_rmse_u = rmse_u_matrix[np.isfinite(rmse_u_matrix)]
    valid_rmse_v = rmse_v_matrix[np.isfinite(rmse_v_matrix)]
    if len(valid_rmse_u) > 0 and len(valid_rmse_v) > 0:
        rmse_max = max(valid_rmse_u.max(), valid_rmse_v.max())
    else:
        rmse_max = 1

    # Compute min/max for Bias
    valid_bias_u = bias_u_matrix[np.isfinite(bias_u_matrix)]
    valid_bias_v = bias_v_matrix[np.isfinite(bias_v_matrix)]
    if len(valid_bias_u) > 0 and len(valid_bias_v) > 0:
        bias_min = min(valid_bias_u.min(), valid_bias_v.min())
        bias_max = max(valid_bias_u.max(), valid_bias_v.max())
    else:
        bias_min, bias_max = -1, 1

    # Compute min/max for Angle Difference
    valid_angle = angle_diff_matrix[np.isfinite(angle_diff_matrix)]
    if len(valid_angle) > 0:
        angle_min = valid_angle.min()
        angle_max = valid_angle.max()
    else:
        angle_min, angle_max = 0, 180

    # Compute min/max for Magnitude Difference
    valid_mag_diff = magnitude_diff_matrix[np.isfinite(magnitude_diff_matrix)]
    if len(valid_mag_diff) > 0:
        mag_min = valid_mag_diff.min()
        mag_max = valid_mag_diff.max()
    else:
        mag_min, mag_max = 0, 1  # Since it's absolute difference, minimum will be >=0

    if len(pressure_levels) > 1:
        dp = (pressure_levels[-1] - pressure_levels[0]) / (len(pressure_levels) - 1)
    else:
        dp = 1

    y_min = pressure_levels[0] - dp / 2
    y_max = pressure_levels[-1] + dp / 2
    x_min = mdates.date2num(all_timestamps[0])
    x_max = mdates.date2num(all_timestamps[-1])

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'RMSE, Bias, Angle Diff, and Magnitude Diff for {title_region}')

    # RMSE U
    ax = axes[0, 0]
    ax.set_title('Weighted RMSE - U')
    im = ax.imshow(rmse_u_matrix, aspect='auto', cmap='viridis', origin='lower',
                   extent=[x_min, x_max, y_min, y_max], vmin=0, vmax=rmse_max)
    plt.colorbar(im, ax=ax, label='RMSE')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_yticks(pressure_levels)
    ax.set_yticklabels(pressure_levels)

    # RMSE V
    ax = axes[0, 1]
    ax.set_title('Weighted RMSE - V')
    im = ax.imshow(rmse_v_matrix, aspect='auto', cmap='viridis', origin='lower',
                   extent=[x_min, x_max, y_min, y_max], vmin=0, vmax=rmse_max)
    plt.colorbar(im, ax=ax, label='RMSE')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_yticks(pressure_levels)
    ax.set_yticklabels(pressure_levels)

    # Bias U
    ax = axes[1, 0]
    ax.set_title('Wind Speed Bias - U')
    im = ax.imshow(bias_u_matrix, aspect='auto', cmap='seismic', origin='lower',
                   extent=[x_min, x_max, y_min, y_max], vmin=bias_min, vmax=bias_max)
    plt.colorbar(im, ax=ax, label='Bias')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_yticks(pressure_levels)
    ax.set_yticklabels(pressure_levels)

    # Bias V
    ax = axes[1, 1]
    ax.set_title('Wind Speed Bias - V')
    im = ax.imshow(bias_v_matrix, aspect='auto', cmap='seismic', origin='lower',
                   extent=[x_min, x_max, y_min, y_max], vmin=bias_min, vmax=bias_max)
    plt.colorbar(im, ax=ax, label='Bias')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_yticks(pressure_levels)
    ax.set_yticklabels(pressure_levels)

    # Angle Difference
    ax = axes[2, 0]
    ax.set_title('Wind Direction Angle Difference')
    im = ax.imshow(angle_diff_matrix, aspect='auto', cmap='magma', origin='lower',
                   extent=[x_min, x_max, y_min, y_max], vmin=angle_min, vmax=angle_max)
    plt.colorbar(im, ax=ax, label='Angle Diff (deg)')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_yticks(pressure_levels)
    ax.set_yticklabels(pressure_levels)
    ax.set_xlabel('Date')

    # Magnitude Difference (Absolute)
    ax = axes[2, 1]
    ax.set_title('Magnitude Difference (|SYNTH - ERA5|)')
    im = ax.imshow(magnitude_diff_matrix, aspect='auto', cmap='coolwarm', origin='lower',
                   extent=[x_min, x_max, y_min, y_max], vmin=mag_min, vmax=mag_max)
    plt.colorbar(im, ax=ax, label='Mag Diff (m/s)')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_yticks(pressure_levels)
    ax.set_yticklabels(pressure_levels)
    ax.set_xlabel('Date')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def compute_one_month_radiosonde(synth_filename, era5_filename, start_month, radiosonde_locations):
    start_time = pd.to_datetime(f"{start_month}-01")
    end_time = start_time + MonthEnd(1)

    results = compute_metrics_for_period(
        synth_filenames=[synth_filename],
        era5_filenames=[era5_filename],
        start_times=[start_time],
        end_times=[end_time],
        radiosonde_locations=radiosonde_locations,
        region_mode=False
    )

    all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix, angle_diff_matrix, magnitude_diff_matrix = results
    plot_results(all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix,
                 angle_diff_matrix, magnitude_diff_matrix, region_mode=False)


def compute_one_year_radiosonde(synth_filenames, era5_filenames, year, radiosonde_locations):
    start_times = [pd.to_datetime(f"{year}-{m:02d}-01") for m in range(1, 13)]
    end_times = [st + MonthEnd(1) for st in start_times]

    results = compute_metrics_for_period(
        synth_filenames=synth_filenames,
        era5_filenames=era5_filenames,
        start_times=start_times,
        end_times=end_times,
        radiosonde_locations=radiosonde_locations,
        region_mode=False
    )

    all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix, angle_diff_matrix, magnitude_diff_matrix = results
    plot_results(all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix,
                 angle_diff_matrix, magnitude_diff_matrix, region_mode=False)


def compute_one_month_region(synth_filename, era5_filename, start_month):
    start_time = pd.to_datetime(f"{start_month}-01")
    end_time = start_time + MonthEnd(1)

    results = compute_metrics_for_period(
        synth_filenames=[synth_filename],
        era5_filenames=[era5_filename],
        start_times=[start_time],
        end_times=[end_time],
        radiosonde_locations=None,
        region_mode=True
    )

    all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix, angle_diff_matrix, magnitude_diff_matrix = results
    plot_results(all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix,
                 angle_diff_matrix, magnitude_diff_matrix, region_mode=True)


def compute_one_year_region(synth_filenames, era5_filenames, year):
    start_times = [pd.to_datetime(f"{year}-{m:02d}-01") for m in range(1, 13)]
    end_times = [st + MonthEnd(1) for st in start_times]

    results = compute_metrics_for_period(
        synth_filenames=synth_filenames,
        era5_filenames=era5_filenames,
        start_times=start_times,
        end_times=end_times,
        radiosonde_locations=None,
        region_mode=True
    )

    all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix, angle_diff_matrix, magnitude_diff_matrix = results
    plot_results(all_timestamps, pressure_levels, rmse_u_matrix, rmse_v_matrix, bias_u_matrix, bias_v_matrix,
                 angle_diff_matrix, magnitude_diff_matrix, region_mode=True)


if __name__ == "__main__":
    #synth_filename = "SYNTH-Oct-2023-SEA-UPDATED.nc"
    #era5_filename = "optimized_ERA5-H2-2023-SEA.nc"

    synth_filename = "SYNTH-Jan-2020-WH.nc"
    era5_filename = "optimized_ERA5-2020-WH.nc"

    synth_filenames = [
        "SYNTH-Jan-2023-USA-UPDATED.nc", "SYNTH-Feb-2023-USA-UPDATED.nc", "SYNTH-Mar-2023-USA-UPDATED.nc",
        #None, None, None,
        #None, None, None,
        "SYNTH-Apr-2023-USA-UPDATED.nc", "SYNTH-May-2023-USA-UPDATED.nc", "SYNTH-Jun-2023-USA-UPDATED.nc",
        "SYNTH-Jul-2023-USA-UPDATED.nc", "SYNTH-Aug-2023-USA-UPDATED.nc", "SYNTH-Sep-2023-USA-UPDATED.nc",
        "SYNTH-Oct-2023-USA-UPDATED.nc", "SYNTH-Nov-2023-USA-UPDATED.nc", "SYNTH-Dec-2023-USA-UPDATED.nc"
    ]
    era5_filenames = ["ERA5-H1-2023-USA.nc", "ERA5-H2-2023-USA.nc"]
    #'''
    synth_filenames = [
        "SYNTH-Jan-2023-SEA-UPDATED.nc", "SYNTH-Feb-2023-SEA-UPDATED.nc", "SYNTH-Mar-2023-SEA-UPDATED.nc",
        "SYNTH-Apr-2023-SEA-UPDATED.nc", "SYNTH-May-2023-SEA-UPDATED.nc", "SYNTH-Jun-2023-SEA-UPDATED.nc",
        "SYNTH-Jul-2023-SEA-UPDATED.nc", "SYNTH-Aug-2023-SEA-UPDATED.nc", "SYNTH-Sep-2023-SEA-UPDATED.nc",
        "SYNTH-Oct-2023-SEA-UPDATED.nc", "SYNTH-Nov-2023-SEA-UPDATED.nc", "SYNTH-Dec-2023-SEA-UPDATED.nc"
    ]
    era5_filenames = ["optimized_ERA5-H1-2023-SEA.nc", "optimized_ERA5-H2-2023-SEA.nc"]
    #'''

    era5_filenames = ["optimized_ERA5-H1-2023-SEA.nc", "optimized_ERA5-H2-2023-SEA.nc"]
    # '''
    synth_filenames = [
        "SYNTH-Jan-2020-WH.nc", "SYNTH-Feb-2020-WH.nc", "SYNTH-Mar-2020-WH.nc",
        None,None,None,
        None,None,None,
        None,None,None,
        #"SYNTH-Apr-2023-SEA-UPDATED.nc", "SYNTH-May-2023-SEA-UPDATED.nc", "SYNTH-Jun-2023-SEA-UPDATED.nc",
        #"SYNTH-Jul-2023-SEA-UPDATED.nc", "SYNTH-Aug-2023-SEA-UPDATED.nc", "SYNTH-Sep-2023-SEA-UPDATED.nc",
        #"SYNTH-Oct-2023-SEA-UPDATED.nc", "SYNTH-Nov-2023-SEA-UPDATED.nc", "SYNTH-Dec-2023-SEA-UPDATED.nc"
    ]
    era5_filenames = ["optimized_ERA5-2020-WH.nc", "optimized_ERA5-2020-WH.nc"]

    start_month = "2020-01"
    year = "2020"

    radiosonde_locations = extract_radiosonde_locations('North_America.csv')
    #print(f"Extracted radiosonde locations:\n{radiosonde_locations}")

    # Example calls:
    #compute_one_month_radiosonde(synth_filename, era5_filename, start_month, radiosonde_locations)
    #compute_one_year_radiosonde(synth_filenames, era5_filenames, year, radiosonde_locations)
    compute_one_year_region(synth_filenames, era5_filenames, year)
    #compute_one_month_region(synth_filename, era5_filename, start_month)
