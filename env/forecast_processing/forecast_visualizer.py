"""
Create 3D visualizations of ERA5 or Synth forecast with colored quiver plots.  Can shoose between
coloring via speed or direction (default is direction).

Also gives an example of doing a side by side comparison of levels between ERA5 and Synth

As well as how to make GIFs
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from metpy.calc import pressure_to_height_std
from metpy.units import units
from utils.Custom3DQuiver import Custom3DQuiver
from matplotlib.projections import register_projection
from termcolor import colored
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast_Subset
from utils.initialize_forecast import initialize_forecasts
import imageio
import copy

class ForecastVisualizer:
    """
    Visualizes forecast data for 3D quiver plots and comparisons.

    Attributes:
        forecast_subset (Forecast_Subset): The forecast subset to visualize.
        render_style (str): Visualization style ('direction' or 'speed').
        pressure_levels (list): Pressure levels in the forecast subset.
        alts2 (numpy.ndarray): Altitudes corresponding to pressure levels.
        flow_field (numpy.ndarray): Processed 3D flow field data.
        levels (numpy.ndarray): Altitude levels for plotting.
    """
    def __init__(self, forecast, render_style = "direction"):
        """
        Initialize the ForecastVisualizer with a forecast subset.

        Args:
            forecast (Forecast_Subset): Subset of the forecast to visualize.
            render_style (str, optional): Style for rendering ('direction' or 'speed').

        Raises:
            Exception: If the provided forecast is not an instance of Forecast_Subset.
        """
        if not isinstance(forecast, Forecast_Subset):
            raise Exception (colored("Provided forecast type is not <class 'era5.forecast.Forecast_Subset'> and instead " + str(type(forecast)),"red"))

        self.render_style = render_style

        self.forecast_subset = forecast
        self.pressure_levels = self.forecast_subset.ds["level"].values.tolist()
        self.pressure_levels = [round(x, 1) for x in self.pressure_levels]
        register_projection(Custom3DQuiver)

    def map_pres2alt(self):
        """
        Map pressure levels to altitudes using standard atmospheric conditions. (Rough approximation)

        Returns:
            list: List of altitudes corresponding to the pressure levels.
        """
        alts = []

        for pres in self.pressure_levels:
            # Define the pressure in Pascals
            pres = pres * units.millibar  # Example pressure value

            # Convert pressure to altitude
            alt = pressure_to_height_std(pres).magnitude
            alts.append(alt)

        return alts

    def generate_flow_array(self, timestamp):
        """
        Generate a 3D flow field array for visualization.

        Args:
            timestamp (numpy.datetime64): Timestamp for the forecast data.
        """

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


    def visualize_3d_planar_flow(self, ax, quiver_skip=1, altitude_quiver_skip=3, show_cbar = False, arrow_head_angle = 84.9, length = .05, arrow_length_ratio=3.5):
        """
        Visualize the 3D planar flow field using quiver plots.

        Args:
            ax (matplotlib.axes._axes.Axes): Axes object for plotting.
            quiver_skip (int, optional): Skip factor for quiver points in x and y.
            altitude_quiver_skip (int, optional): Skip factor for altitude levels.
            show_cbar (bool, optional): Whether to display the colorbar.
            arrow_head_angle (float, optional): Angle of arrowheads in quiver plot.
            length (float, optional): Length of arrows.
            arrow_length_ratio (float, optional): Ratio of arrowhead length to arrow length.
        """

        #For altitude quiver skipping
        for z in range(0, self.flow_field.shape[0]):
            if z % altitude_quiver_skip != 0:
                continue

            # UPDATE added indexing 'ij' which fixed a hidden bug in visualization when flows are not all the same magnitude.
            X, Y = np.meshgrid(np.arange(self.flow_field.shape[1]), np.arange(self.flow_field.shape[2]), indexing='ij')
            U = self.flow_field[z, :, :, 0]
            V = self.flow_field[z, :, :, 1]
            W = self.flow_field[z, :, :, 2]  # Flow is only in the X-Y plane

            Z = np.full_like(X, self.levels[z])

            # Calculate directions for color mapping
            directions = np.arctan2(V, U)
            speed = np.sqrt(V**2 + U**2)
            res = 1

            # For Speed
            if self.render_style == "speed":
                norm = plt.Normalize(0, 50)
                colors = cm.rainbow(norm(speed))

            elif self.render_style == "direction":
                norm = plt.Normalize(-np.pi, np.pi)
                colors = cm.hsv(norm(directions)) #for Directions

            else:
                raise Exception("Undefined render_style for Forecast Visualization")

            for i in range(0, X.shape[0], quiver_skip):
                for j in range(0, Y.shape[1], quiver_skip):
                    # Skip z quivers based on the altitude (z) level
                    #if (i + j) % altitude_quiver_skip != 0:  # Skip based on custom condition
                    #    continue

                    ax.quiver(X[i, j] / res, Y[i, j] / res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                              length = length, arrow_length_ratio=arrow_length_ratio, color=colors[i, j], arrow_head_angle=arrow_head_angle)
                              #length = .1, arrow_length_ratio = 1.5, color = colors[i, j], arrow_head_angle = 75)
                              #length = 1, arrow_length_ratio = .5, color = colors[i, j], arrow_head_angle = 75, normalize = False)

                    #ax.quiver(X[i, j] / res, Y[i, j] / res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                    #          length=.25, arrow_length_ratio=1.5, color=colors[i, j])

        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        ax.set_zlabel('Pressure Level (mb)')


        if self.render_style == "direction":
            colormap = plt.colormaps.get_cmap('hsv')
            # colors = colormap(scaled_z)
            sm = plt.cm.ScalarMappable(cmap=colormap)
            sm.set_clim(vmin=-3.14, vmax=3.14)

            if show_cbar:
                cbar = plt.colorbar(sm, ax=ax, shrink=.4, pad=.15)
                cbar.set_label('Wind Direction (radians')


        if self.render_style == "speed":
            mappable = cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
            mappable.set_array(speed)

            if show_cbar:
                cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
                cbar.set_label('Wind Speed')


        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Setting custom tick labels without changing the plot bounds

        ax.set_xticks(np.linspace(x_min, x_max, 6), np.linspace(self.forecast_subset.ds.longitude[0].values, self.forecast_subset.ds.longitude[-1].values, 6, dtype=float))
        ax.set_yticks(np.linspace(y_min, y_max, 6), np.linspace(self.forecast_subset.ds.latitude[0].values, self.forecast_subset.ds.latitude[-1].values, 6, dtype=float))

        plt.title(self.timestamp)


        if self.forecast_subset.Forecast.forecast_type == "SYNTH":
            ax.set_zticks(self.alts2[::5])
            ax.set_zticklabels(self.pressure_levels[::5])
            plt.title("Synthetic Forecast")

        if self.forecast_subset.Forecast.forecast_type == "ERA5":
            ax.set_zticks(self.alts2)
            ax.set_zticklabels(self.pressure_levels)
            plt.title("ERA5 Forecast")

        #plt.title()


def plot_3d_quiver(timestamp, forecast_subset,  quiver_skip = 2):
    forecast_visualizer = ForecastVisualizer(forecast_subset)
    forecast_visualizer.generate_flow_array(timestamp=timestamp)

    # Initialize Figure
    fig = plt.figure(figsize=(15, 10))
    # ax1 = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(111, projection='custom3dquiver')
    # Manually add a CustomAxes3D to the figure
    # ax1 = Custom3DQuiver(fig)
    fig.add_axes(ax1)

    print("Plotting Forecast (" +  forecast_subset.Forecast.forecast_type + ") -  "  + str(timestamp))
    forecast_visualizer.visualize_3d_planar_flow(ax1, quiver_skip=quiver_skip)
    # plt.savefig(str(i) +'.png')
    plt.show()
    #plt.close()


def plot_side_by_side_levels():
    """
    Shows an example of creating a gif of comparing individual pressure levels between Synth and ERA5 for the same date

    ERA5 and Synth need to contain the same downloaded region and date windows.

    We use xarrays isel method to find the matching pressure levels (since Synth is altitude based, and ERA5 is pressure
    based)
    """

    # Now let's make a side by side GIF
    # Leaving off 20 hpa and 150 hpa since synthwinds doesn't include those
    for i in range(1, forecast_subset_era5.level_dim - 1):
        #Get current ERA5 pressure level
        pres = forecast_subset_era5.ds.level.values[i]
        print(pres)

        # Find altitude for pressure level
        alt_era5 = forecast_subset_era5.get_alt_from_pressure(pres)

        # For Synthwinds, can Assume every coordinate has the same altitude column, so just take first index
        alt_column = forecast_subset_synth.ds.isel(time=0, latitude=0, longitude=0)['z'].values / 9.81
        #print(alt_column)


        # Find the index of the nearest z value
        nearest_idx = np.argmin(np.abs(alt_column - alt_era5))
        print(pres, alt_era5,
              forecast_subset_synth.ds.isel(time=0, latitude=0, level=nearest_idx, longitude=0)['z'].values / 9.81)

        # Take some slices of the forecasts for plotting
        era_5_slice = copy.deepcopy(forecast_subset_era5)
        era_5_slice.ds = forecast_subset_era5.ds.isel(level=slice(i, i + 1))
        forecast_visualizer_era5 = ForecastVisualizer(era_5_slice)

        synth_slice = copy.deepcopy(forecast_subset_synth)
        synth_slice.ds = forecast_subset_synth.ds.isel(level=slice(nearest_idx, nearest_idx + 1))
        forecast_visualizer_synth = ForecastVisualizer(synth_slice)

        # Plot Side by Side

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(nrows=1, ncols=2)
        ax1 = fig.add_subplot(gs[0, 0], projection='custom3dquiver')
        ax2 = fig.add_subplot(gs[0, 1], projection='custom3dquiver')

        fig.add_axes(ax1)
        forecast_visualizer_era5.generate_flow_array(timestamp=timestamp)
        forecast_visualizer_era5.visualize_3d_planar_flow(ax1, quiver_skip = 2)

        fig.add_axes(ax2)
        forecast_visualizer_synth.generate_flow_array(timestamp=timestamp)
        forecast_visualizer_synth.visualize_3d_planar_flow(ax2, quiver_skip = 2)

        plt.show()


if __name__ == '__main__':
    # Import Forecasts
    FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

    #env_params["rel_dist"] = 100_000_000 # Manually Override relative distance to show a whole subset
    #timestamp = "2023-01-03T00:00:00.000000000"

    if env_params["seed"] != None:
        # print("Seed", seed)
        np_rng = np.random.default_rng(env_params["seed"])
    else:
        np_rng = np.random.default_rng(np.random.randint(0, 2 ** 32))

    # Assign central coordinate for synth
    #forecast_subset_synth.assign_coord(
    #    0.5 * (forecast_subset_synth.Forecast.LAT_MAX + forecast_subset_synth.Forecast.LAT_MIN),
    #    0.5 * (forecast_subset_synth.Forecast.LON_MAX + forecast_subset_synth.Forecast.LON_MIN),
    #    timestamp)

    forecast_subset_synth.randomize_coord(np_rng)

    # Assign samecoordinate for era5
    forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)
    forecast_subset_era5.assign_coord(forecast_subset_synth.lat_central, forecast_subset_synth.lon_central, forecast_subset_synth.start_time)

    forecast_subset_era5.subset_forecast( days=1)
    forecast_subset_synth.subset_forecast(days=1)


    print("randomized coord: ", forecast_subset_era5.start_time,
                                forecast_subset_era5.lat_central,
                                forecast_subset_era5.lon_central)
    #forecast_subset_era5.subset_forecast(days=1)


    # ***** PLOTTING EXAMPLES *******

    # Plot ERA5 or Synth 3d Quiver Plot
    plot_3d_quiver(timestamp = forecast_subset_era5.start_time, forecast_subset = forecast_subset_era5, quiver_skip = 5)

    # Plot ERA5 and Synth side by side levels
    plot_side_by_side_levels()


    #Generate gif of flowfield
    #with imageio.get_writer('Synth-wind.gif', mode='I', duration=500, loop = 0) as writer:
    #    for i in range(len(forecast_visualizer.forecast_subset.ds.time.values)):
    #        image = imageio.imread(str(i) +'.png')
    #        writer.append_data(image)


