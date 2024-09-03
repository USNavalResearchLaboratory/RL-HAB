from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from metpy.calc import pressure_to_height_std
from metpy.units import units
from utils.Custom3DQuiver import Custom3DQuiver
from matplotlib.projections import register_projection
from termcolor import colored
from env3d.config.env_config import env_params
import imageio
from era5.forecast import Forecast, Forecast_Subset


class ForecastVisualizer:
    """
    Visualizes a forecast.  a Forecast needs to be converted to a Forecast_Subset before visualizing to determine ranges
    and convert to a numpy array for faster processing.
    """
    def __init__(self, forecast):
        if not isinstance(forecast, Forecast_Subset):
            raise Exception (colored("Provided forecast type is not <class 'era5.forecast.Forecast_Subset'> and instead " + str(type(forecast)),"red"))


        self.forecast_subset = forecast
        self.pressure_levels = self.forecast_subset.ds["level"].values
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
        self.alts2 = self.forecast_subset.ds.sel(latitude=self.forecast_subset.lat_central, longitude=self.forecast_subset.lon_central,
                                                 time=timestamp, method='nearest')['z'].values * .001
        # Assign the new altitude coordinate
        self.forecast_subset.ds = self.forecast_subset.ds.assign_coords(altitude=('level', self.alts2))

        self.timestamp = timestamp
        time_index = list(self.forecast_subset.ds.time.values).index(self.forecast_subset.ds.sel(time=self.timestamp, method='nearest').time)

        # ERA5 data variables structure that we download is (time, level, latitude, longitude)
        self.u = self.forecast_subset.ds['u'][time_index, :, :, :].data #should this be values
        self.v = self.forecast_subset.ds['v'][time_index, :, :, :].values
        self.z = self.forecast_subset.ds['z'][time_index, :, :, :].values

        # need to reshape array to level, Longitude(X), latitude(Y) for plotting.
        self.u = np.swapaxes(self.u, 1, 2)
        self.v = np.swapaxes(self.v, 1, 2)
        self.z = np.swapaxes(self.z, 1, 2)
        self.w = np.zeros_like(self.u)

        self.levels = self.forecast_subset.ds['altitude']

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
            speed = np.sqrt(V**2 + U**2)

            # For Speed
            #norm = plt.Normalize(np.min(speed), np.max(speed))
            #colors = cm.hsv(norm(speed))

            # For Direction
            norm = plt.Normalize(-np.pi, np.pi)
            colors = cm.hsv(norm(directions)) #for Directions
            res = 1

            for i in range(0, X.shape[0], skip):
                for j in range(0, Y.shape[1], skip):
                    ax.quiver(X[i, j] / res, Y[i, j] / res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                              length = .05, arrow_length_ratio=3.5, color=colors[i, j], arrow_head_angle=84.9)
                              #length = .1, arrow_length_ratio = 1.5, color = colors[i, j], arrow_head_angle = 75)
                              #length = 1, arrow_length_ratio = .5, color = colors[i, j], arrow_head_angle = 75, normalize = False)

                    #ax.quiver(X[i, j] / res, Y[i, j] / res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                    #          length=.25, arrow_length_ratio=1.5, color=colors[i, j])

        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_zlabel('Pressure Level (mb)')

        #'''
        #Color Direction
        colormap = plt.colormaps.get_cmap('hsv')
        # colors = colormap(scaled_z)
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_clim(vmin=-3.14, vmax=3.14)
        plt.colorbar(sm, ax=ax, shrink=.8, pad=.025)
        #'''

        '''
        #Color Speed
        mappable = cm.ScalarMappable(cmap=cm.hsv, norm=norm)
        mappable.set_array(speed)
        cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
        cbar.set_label('Wind Speed')
        '''

        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Setting custom tick labels without changing the plot bounds
        ax.set_zticks(self.alts2)
        ax.set_zticklabels(self.pressure_levels)
        ax.set_xticks(np.linspace(x_min, x_max, 6), np.linspace(self.forecast_subset.ds.longitude[0].values, self.forecast_subset.ds.longitude[-1].values, 6, dtype=float))
        ax.set_yticks(np.linspace(y_min, y_max, 6), np.linspace(self.forecast_subset.ds.latitude[0].values, self.forecast_subset.ds.latitude[-1].values, 6, dtype=float))

        plt.title(self.timestamp)


if __name__ == '__main__':
    #filename = "SHAB14V_ERA5_20220822_20220823.nc"
    filename = "SYNTH-Jan-2023-SEA.nc"
    #filename = "Jan-2023-SEA.nc"
    FORECAST_PRIMARY = Forecast(filename)

    env_params["rel_dist"] = 10_000_000 #does this override work?

    print(FORECAST_PRIMARY.ds_original)

    forecast_subset = Forecast_Subset(FORECAST_PRIMARY)
    forecast_subset.assign_coord(0.5 * (forecast_subset.Forecast.LAT_MAX + forecast_subset.Forecast.LAT_MIN),
                                 0.5 * (forecast_subset.Forecast.LON_MAX + forecast_subset.Forecast.LON_MIN),
                                 "2023-01-01T00:00:00.000000000")
    #forecast_subset.randomize_coord()
    forecast_subset.subset_forecast(days=1)

    #forecast_subset.ds = forecast_subset.ds.isel(level = slice(1,2))

    print(forecast_subset.ds['z'].values/9.81)

    print(forecast_subset.ds)

    #sdfsd

    # Analyze Data
    forecast_visualizer = ForecastVisualizer(forecast_subset)
    skip = 10


    i = 0
    for timestamp in forecast_visualizer.forecast_subset.ds.time.values:
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
        #plt.savefig(str(i) +'.png')
        plt.show()
        plt.close()
        i +=1

    #plt.show()

    #Generate gif of flowfield
    with imageio.get_writer('Synth-wind.gif', mode='I', duration=500, loop = 0) as writer:
        for i in range(len(forecast_visualizer.forecast_subset.ds.time.values)):
            image = imageio.imread(str(i) +'.png')
            writer.append_data(image)


