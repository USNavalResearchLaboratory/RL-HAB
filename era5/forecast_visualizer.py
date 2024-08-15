from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from metpy.calc import pressure_to_height_std
from metpy.units import units
from utils.Custom3DQuiver import Custom3DQuiver
from matplotlib.projections import register_projection

from env3d.config.env_config import env_params
from era5 import config_earth

from era5 import ERA5
import imageio
from era5.forecast import Forecast, Forecast_Subset


class ForecastVisualizer:
    def __init__(self, forecast):

        self.forecast = forecast
        self.pressure_levels = self.forecast.ds["level"].values
        '''
        self.ds = forecast.ds

        self.pressure_levels = self.ds["level"].values

        # ERA5 stuff
        coord = config_earth.simulation['start_coord']
        self.gfs = ERA5.ERA5(coord)

        self.start_time = config_earth.simulation['start_time']
        self.alts2 = self.ds.sel(latitude=coord['lat'], longitude=coord['lon'], time = self.start_time, method='nearest')['z'].values*.001

        # Assign the new altitude coordinate
        self.ds = self.ds.assign_coords(altitude=('level', self.alts2))
        '''


        register_projection(Custom3DQuiver)

    def map_pres2alt(self):
        # Use interpolation to transform the original Z values to the desired visual scale
        alts = []

        for pres in self.pressure_levels:
            # Define the pressure in Pascals
            pres = pres * units.millibar  # Example pressure value

            # Convert pressure to altitude
            alt = pressure_to_height_std(pres).magnitude
            alts.append(alt)

        return alts

    def generate_flow_array(self, timestamp):

        #For plotting altitude levels
        self.alts2 = self.forecast.ds.sel(latitude=self.forecast.lat_central, longitude=self.forecast.lon_central,
                                          time=timestamp, method='nearest')['z'].values * .001
        # Assign the new altitude coordinate
        self.forecast.ds = self.forecast.ds.assign_coords(altitude=('level', self.alts2))

        self.timestamp = timestamp
        time_index = list(self.forecast.ds.time.values).index(self.forecast.ds.sel(time=self.timestamp, method='nearest').time)

        # ERA5 data variables structure that we download is (time, level, latitude, longitude)
        self.u = self.forecast.ds['u'][time_index, :, :, :].data
        self.v = self.forecast.ds['v'][time_index, :, :, :].values
        self.z = self.forecast.ds['z'][time_index, :, :, :].values

        # need to reshape array to level, Longitude(X), latitude(Y) for plotting.
        self.u = np.swapaxes(self.u, 1, 2)
        self.v = np.swapaxes(self.v, 1, 2)
        self.z = np.swapaxes(self.z, 1, 2)
        self.w = np.zeros_like(self.u)

        self.levels = self.forecast.ds['altitude']

        self.flow_field = np.stack([self.u, self.v, self.w, self.z], axis=-1)


    def visualize_3d_planar_flow(self, ax, skip=1):
        '''
        Plot the Flow Field
        '''
        for z in range(self.flow_field.shape[0]):
            # UPDATE added indexing 'ij' which fixed a hidden bug in visualization when flows are not all the same magnitude.
            X, Y = np.meshgrid(np.arange(self.flow_field.shape[1]), np.arange(self.flow_field.shape[2]), indexing='ij')
            U = self.flow_field[z, :, :, 0]
            V = self.flow_field[z, :, :, 1]
            W = self.flow_field[z, :, :, 2]  # Flow is only in the X-Y plane

            Z = np.full_like(X, self.levels[z])

            # Calculate directions for color mapping
            directions = np.arctan2(V, U)
            norm = plt.Normalize(-np.pi, np.pi)
            colors = cm.hsv(norm(directions))
            res = 1

            for i in range(0, X.shape[0], skip):
                for j in range(0, Y.shape[1], skip):
                    ax.quiver(X[i, j] / res, Y[i, j] / res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                              length = .1, arrow_length_ratio=1.5, color=colors[i, j], arrow_head_angle=75)

                    #ax.quiver(X[i, j] / res, Y[i, j] / res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                    #          length=.25, arrow_length_ratio=1.5, color=colors[i, j])

        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_zlabel('Pressure Level (mb)')


        colormap = plt.colormaps.get_cmap('hsv')
        # colors = colormap(scaled_z)
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_clim(vmin=-3.14, vmax=3.14)
        #plt.colorbar(sm, ax=ax, shrink=.8, pad=.025)

        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Setting custom tick labels without changing the plot bounds
        ax.set_zticks(self.alts2)
        ax.set_zticklabels(self.pressure_levels)

        ax.set_xticks(np.linspace(x_min, x_max, 6), np.linspace(self.forecast.ds.longitude[0].values, self.forecast.ds.longitude[-1].values, 6, dtype=float))
        ax.set_yticks(np.linspace(y_min, y_max, 6), np.linspace(self.forecast.ds.latitude[0].values, self.forecast.ds.latitude[-1].values, 6, dtype=float))

        plt.title(self.timestamp)


if __name__ == '__main__':

    #register_projection(Custom3DQuiver)

    pres_min = env_params['pres_min']
    pres_max = env_params['pres_max']
    rel_dist = env_params['rel_dist']

    filename = "SHAB14V_ERA5_20220822_20220823.nc"
    FORECAST_PRIMARY = Forecast(filename)

    forecast_subset = Forecast_Subset(FORECAST_PRIMARY)
    forecast_subset.randomize_coord()
    print("random_coord", forecast_subset.lat_central, forecast_subset.lon_central, forecast_subset.start_time)
    forecast_subset.subset_forecast(rel_dist, pres_min, pres_max)

    # Analyze Data
    forecast_visualizer = ForecastVisualizer(forecast_subset)
    skip = 2

    print(forecast_visualizer.forecast.ds.time.values)

    i = 0
    for timestamp in forecast_visualizer.forecast.ds.time.values:
        #forecast.randomize_coord()
        #forecast.subset_forecast(rel_dist, pres_min, pres_max)

        forecast_visualizer.generate_flow_array(timestamp = timestamp)

        # Initialize Figure
        fig = plt.figure(figsize=(15, 10))
        #ax1 = fig.add_subplot(111, projection='3d')
        ax1 = fig.add_subplot(111, projection='custom3dquiver')
        # Manually add a CustomAxes3D to the figure
        #ax1 = Custom3DQuiver(fig)
        fig.add_axes(ax1)

        print("Saving Figure " + str(timestamp))
        forecast_visualizer.visualize_3d_planar_flow(ax1, skip)
        plt.savefig(str(i) +'.png')
        #plt.show()
        plt.close()
        i +=1

    #plt.show()

    #Generate gif of flowfield
    with imageio.get_writer('wind3.gif', mode='I') as writer:
        for i in range(len(forecast_visualizer.forecast.ds.time.values)):
            image = imageio.imread(str(i) +'.png')
            writer.append_data(image)


