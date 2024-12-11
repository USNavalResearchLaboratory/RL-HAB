"""
Simple Algorithm to classify forecasts based on number of opposing winds.  Similar to method used in RadioWinds.

For now, we are assuming looking at 6 hour increments for a 24 hour window  (4 time instances)

For each time instance,  calculate opposing winds with **n_sectors** (default is 8)

Levels are based on the netcdf file and pressure range.  (Synth will have more levels than ERA5)
"""


import numpy as np
from env.forecast_processing.forecast import Forecast, Forecast_Subset
import windrose
import pandas as pd
from env.forecast_processing.forecast_visualizer import ForecastVisualizer
import matplotlib.pyplot as plt
from utils.initialize_forecast import initialize_forecasts


class ForecastClassifier:
    """
    Classifies forecasts based on opposing wind patterns.

    Methods:
        - determine_opposing_winds: Identify opposing wind levels and directions.
        - determine_OW_Rate: Calculate the opposing wind rate for a forecast subset.
    """
    def __init__(self):
        """
        Initialize the ForecastClassifier.
        """
        pass

    def determine_opposing_winds(self, wd, levels, n_sectors):
        """
        Identify opposing wind levels and directions from wind data.

        Args:
            wd (numpy.ndarray): Wind direction array (degrees).
            levels (numpy.ndarray): Pressure or altitude levels.
            n_sectors (int): Number of angular sectors.

        Returns:
            tuple:
                - opposing_wind_directions (numpy.ndarray): Indices of opposing wind directions.
                - opposing_wind_levels (numpy.ndarray): Levels with opposing winds.
        """

        dir_edges, var_bins, table = windrose.windrose.histogram(wd, levels, bins=levels, nsector=n_sectors)

        #Determine the sectors (directions) that contain non zero values (altitude levels that have wind)
        df = pd.DataFrame(table)

        altitude_lookup_idxs = df.apply(np.flatnonzero, axis=0) # altitude can be pressure or height, depending on by_pressure variable

        opposing_wind_levels = np.array([])
        opposing_wind_directions = np.array([])

        # Determine the sectors that have opposing winds by checking the current index and the complimentary index at n_sectors/2.
        # Also determine the altitudes contains in the opposing wind pairs for calculating probabilities later.
        for i in range (0,int(n_sectors/2)):
            # check if opposing sectors in the histogram tables have values greater than 0
            # (therefore, there are winds in that sectors)
            if np.sum(table, axis=0)[i] != 0 and np.sum(table, axis=0)[i+int(n_sectors/2)] != 0:
                for idx in altitude_lookup_idxs[i]:
                    opposing_wind_levels = np.append(opposing_wind_levels, var_bins[idx])
                    opposing_wind_directions = np.append(opposing_wind_directions, i)
                for idx in altitude_lookup_idxs[i+int(n_sectors/2)]:
                    #print(var_bins[idx])
                    opposing_wind_levels = np.append(opposing_wind_levels, var_bins[idx])
                    opposing_wind_directions = np.append(opposing_wind_directions, i+int(n_sectors/2))

        # sort the opposing wind altitudes and direction idxs (format later) in ascending order and remove duplicates
        opposing_wind_levels = np.sort(np.unique(opposing_wind_levels))
        opposing_wind_directions = np.sort(np.unique(opposing_wind_directions))

        return opposing_wind_directions, opposing_wind_levels


    def determine_OW_Rate(self, forecast_subset):
        """
        Calculate the opposing wind rate for a forecast subset over a 24-hour window.

        Args:
            forecast_subset (Forecast_Subset): Subset of the forecast.

        Returns:
            tuple:
                - scores (list): Number of opposing wind levels at each time interval.
                - score (float): Normalized opposing wind rate.
        """

        #Assuming 24 hour subset window right now

        start_time = forecast_subset.start_time
        timestamp = start_time
        scores = []

        intervals = 4 # Default Value (how many time incrememnts to look at given a 24 hour window)
        n_sectors = 8 # Default Value (how many angular bins to classify angular bins into)

        for i in range (0,intervals):

            z, u, v = forecast_subset.np_lookup(forecast_subset.lat_central, forecast_subset.lon_central,
                                                timestamp)

            bearing, speed = forecast_subset.windVectorToBearing(u, v)
            bearing = bearing % (2 * np.pi)
            bearing = np.degrees(bearing)

            levels = forecast_subset.ds.level.values
            opposing_wind_directions, opposing_wind_levels = self.determine_opposing_winds(bearing, levels, n_sectors)

            scores.append(len(opposing_wind_levels))

            timestamp = timestamp + np.timedelta64(6, "h")


        max_score = (forecast_subset.level_dim*intervals)

        score = np.sum(scores)/max_score

        return scores, score


if __name__ == '__main__':
    # Import Forecasts
    FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

    # Initialize ForecastClassifier
    ForecastClassifier = ForecastClassifier()

    #randomize coord, ERA5 or Synth
    forecast_subset = forecast_subset_synth #choose _era5 or _synth
    forecast_subset.randomize_coord()
    print("random_coord", forecast_subset.lat_central, forecast_subset.lon_central, forecast_subset.start_time)
    forecast_subset.subset_forecast()

    #Determine Forecast Score
    scores, score = ForecastClassifier.determine_OW_Rate(forecast_subset)
    print(scores,score)

    # Visualize the forecast at the first timestamp
    Forecast_visualizer = ForecastVisualizer(forecast_subset)
    Forecast_visualizer.generate_flow_array(timestamp=forecast_subset.start_time)

    # Initialize Figure
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111, projection='custom3dquiver')

    fig.add_axes(ax1)

    Forecast_visualizer.visualize_3d_planar_flow(ax1, quiver_skip= 2)
    plt.show()
