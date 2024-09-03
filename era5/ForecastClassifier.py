import numpy as np
from era5.forecast import Forecast, Forecast_Subset
import windrose
import pandas as pd
from era5.forecast_visualizer import ForecastVisualizer
import matplotlib.pyplot as plt
from env3d.config.env_config import env_params


class ForecastClassifier:
    def __init__(self):
        pass

    def determine_opposing_winds(self, wd, levels, n_sectors):

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

        #Assuming 24 hour subset window right now

        start_time = forecast_subset.start_time
        timestamp = start_time
        scores = []

        intervals = 4

        n_sectors = 4

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
    filename = "July-2024-SEA.nc"
    FORECAST_PRIMARY = Forecast(filename)

    forecast_subset = Forecast_Subset(FORECAST_PRIMARY)
    forecast_subset.randomize_coord()
    #print("random_coord", forecast_subset.lat_central, forecast_subset.lon_central, forecast_subset.start_time)
    forecast_subset.subset_forecast()

    ForecastClassifier = ForecastClassifier()
    scores, score = ForecastClassifier.determine_OW_Rate(forecast_subset)

    print(scores,score)

    timestamp = forecast_subset.start_time
    skip = 2

    Forecast_visualizer = ForecastVisualizer(forecast_subset)
    Forecast_visualizer.generate_flow_array(timestamp=timestamp)

    # Initialize Figure
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111, projection='custom3dquiver')

    fig.add_axes(ax1)

    Forecast_visualizer.visualize_3d_planar_flow(ax1, skip)
    plt.show()
