from matplotlib import cm
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
#from mpl_toolkits.basemap import Basemap

#import ERA5
from era5 import config_earth

from utils.Custom3DQuiver import Custom3DQuiver
from matplotlib.projections import register_projection
register_projection(Custom3DQuiver)
from matplotlib.colors import Normalize, hsv_to_rgb

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import ERA5
import imageio
from forecast import Forecast
import fluids
import pvlib

from metpy.calc import pressure_to_height_std
from metpy.units import units


class ForecastVisualizer:
    def __init__(self):
        pres_min = config_earth.rl_params['pres_min']
        pres_max = config_earth.rl_params['pres_max']
        rel_dist = config_earth.rl_params['rel_dist']

        #No Override of Forecast for now?
        forecast = Forecast(rel_dist, pres_min, pres_max)
        self.ds = forecast.ds

        self.mandatory_pressure_levels = self.ds["level"].values

        # ERA5 stuff
        coord = config_earth.simulation['start_coord']
        self.gfs = ERA5.ERA5(coord)

        self.alts2 = self.ds.sel(latitude=coord['lat'], longitude=coord['lon'], method='nearest').isel(time=0)['z'].values*.001

        # Assign the new altitude coordinate
        self.ds = self.ds.assign_coords(altitude=('level', self.alts2))

        # Drop the old pressure coordinate if you want
        #self.ds = self.ds.swap_dims({'level': 'altitude'})

        print(self.ds)


        register_projection(Custom3DQuiver)

    def map_pres2alt(self):
        # Use interpolation to transform the original Z values to the desired visual scale
        alts = []

        for pres in self.mandatory_pressure_levels:
            # Define the pressure in Pascals
            pres = pres * units.millibar  # Example pressure value

            # Convert pressure to altitude
            alt = pressure_to_height_std(pres).magnitude
            alts.append(alt)

        return alts



    def generate_flow_array(self, time_index):

        self.timestamp = self.ds.time.values[time_index]
        # ERA5 data variables structure that we download is (time, level, latitude, longitude)
        self.u = self.ds['u'][time_index, :, :, :].data
        self.v = self.ds['v'][time_index, :, :, :].values
        self.z = self.ds['z'][time_index, :, :, :].values

        # need to reshape array to level, Longitude(X), latitude(Y) for plotting.
        self.u = np.swapaxes(self.u, 1, 2)
        self.v = np.swapaxes(self.v, 1, 2)
        self.z = np.swapaxes(self.z, 1, 2)
        self.w = np.zeros_like(self.u)

        self.levels = self.ds['altitude']

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

            #Z = np.full_like(X, flow_field[z, 0, 0, 3])
            Z = np.full_like(X, self.levels[z])

            #print(flow_field[z, 0, 0, 3])
            #print(Z)

            #sdfsd

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


        #print(self.ds.longitude[0].values, self.ds.longitude[-1].values)
        #ax.set_xlim(50, 75)
        #ax.set_ylim(ds.latitude[0].values, ds.latitude[-1].values)
        #ax.set_ylim(0, (self.y_dim) / self.res)
        #ax.set_zlim(0, self.z_dim)  # Set the z-axis limit to the maximum altitude

        colormap = plt.colormaps.get_cmap('hsv')
        # colors = colormap(scaled_z)
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_clim(vmin=-3.14, vmax=3.14)
        plt.colorbar(sm, ax=ax, shrink=.8, pad=.025)

        #Change ticks

        # Get current axis limits
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Setting custom tick labels without changing the plot bounds
        #z_ticks = ax.get_zticks()



        #ax.set_zticks(z_transformed[::-1])
        #z_ticks = ax.get_zticks()
        #ax.set_zticks(z_ticks[::-1])

        # Setting custom tick labels without changing the plot bounds
        #z_ticks = ax.get_zticks()
        # print(z_ticks)
        #z_ticks_reversed = z_ticks[::-1]
        #ax.set_zticks(z_ticks_reversed)

        print(self.mandatory_pressure_levels)
        #print(z_transformed)

        # Setting custom tick labels without changing the plot bounds
        z_ticks = ax.get_zticks()
        # print(z_ticks)
        z_ticks_reversed = z_ticks[::-1]
        ax.set_zticks(self.alts2)
        ax.set_zticklabels(self.mandatory_pressure_levels)
        #ax.set_zlim(self.levels[0], self.levels[-1])

        #ax.set_zlim(self.levels[-1], self.levels[0])
        ax.set_xticks(np.linspace(x_min, x_max, 5), np.linspace(self.ds.longitude[0].values, self.ds.longitude[-1].values, 5, dtype=int))
        ax.set_yticks(np.linspace(y_min, y_max, 5), np.linspace(self.ds.latitude[0].values, self.ds.latitude[-1].values, 5, dtype=int))

        #plt.title(self.timestamp)

        '''
        # Customizing tick positions on the x-axis
        new_y_ticks = np.linspace(ds.latitude[-1].values, ds.latitude[0].values, 5)  # Example: ticks at 0, 2, 4, 6, 8, 10
        ax.set_yticks(new_y_ticks)
    
        new_x_ticks = np.linspace(ds.longitude[0].values, ds.longitude[-1].values, 5)  # Example: ticks at 0, 2, 4, 6, 8, 10
        ax.set_xticklabels(new_x_ticks)
        '''

if __name__ == '__main__':

    register_projection(Custom3DQuiver)

    # Analyze Data
    forecast_visualizer = ForecastVisualizer()
    skip = 2


    for i in range(len(forecast_visualizer.ds.time.values)):
        forecast_visualizer.generate_flow_array(time_index=i)

        # Initialize Figure
        fig = plt.figure(figsize=(15, 10))
        #ax1 = fig.add_subplot(111, projection='3d')
        ax1 = fig.add_subplot(111, projection='custom3dquiver')
        # Manually add a CustomAxes3D to the figure
        #ax1 = Custom3DQuiver(fig)
        fig.add_axes(ax1)

        print("Saving Figure " + str(forecast_visualizer.ds.time.values[i]))
        forecast_visualizer.visualize_3d_planar_flow(ax1, skip)
        #plt.savefig(str(i) +'.png')
        plt.show()
        plt.close()

    #plt.show()

    #Generate gif of flowfield
    with imageio.get_writer('wind2.gif', mode='I') as writer:
        for i in range(len(forecast_visualizer.ds.time.values)):
            image = imageio.imread(str(i) +'.png')
            writer.append_data(image)


