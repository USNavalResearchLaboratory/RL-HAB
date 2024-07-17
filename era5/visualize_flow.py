from termcolor import colored
import math
import gmplot
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
#from mpl_toolkits.basemap import Basemap

import ERA5
import config_earth

import plotly.graph_objects as go
import plotly.io as pio




ds = xr.open_dataset("forecasts/" + config_earth.netcdf_era5['filename'])


#for now doing time = 0,  Forecasts not changing in time


#Reverse order of latitude
ds = ds.reindex(latitude=list(reversed(ds.latitude)))


#ERA5 orgnization is time, level, latitude(Y), Longitude(X)
t = 0
u = ds['u'][t,:,:,:].values
v = ds['v'][t,:,:,:].values
z = ds['z'][t,:,:,:].values




#need to reshape array to level, Longitude(X), latitude(Y) for plotting.
u = np.swapaxes(u, 1,2)
v = np.swapaxes(v, 1,2)
z = np.swapaxes(z, 1,2)
w = np.zeros_like(u)

levels = ds['level']

print(levels)



flow_field = np.stack([u, v, w, z], axis=-1)

print(flow_field.shape)


def visualize_3d_planar_flow(ax, skip=1, interpolation_point=None):
    '''
    Plot the Flow Field
    '''
    for z in range(flow_field.shape[0]):
        # UPDATE added indexing 'ij' which fixed a hidden bug in visualization when flows are not all the same magnitude.
        X, Y = np.meshgrid(np.arange(flow_field.shape[1]), np.arange(flow_field.shape[2]), indexing='ij')
        U = flow_field[z, :, :, 0]
        V = flow_field[z, :, :, 1]
        W = flow_field[z, :, :, 2]  # Flow is only in the X-Y plane

        #Z = np.full_like(X, flow_field[z, 0, 0, 3])
        Z = np.full_like(X, levels[z])

        #print(flow_field[z, 0, 0, 3])
        #print(Z)

        #sdfsd

        # Calculate directions for color mapping
        directions = np.arctan2(V, U)
        norm = plt.Normalize(-np.pi, np.pi)
        colors = cm.rainbow(norm(directions))

        for i in range(0, X.shape[0], skip):
            for j in range(0, Y.shape[1], skip):
                ax.quiver(X[i, j] / res, Y[i, j] / res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                          # For the small arena
                          # length=self.magnitudes[z] * .5/self.x_dim, arrow_length_ratio=1/self.x_dim, color=colors[i, j])
                          # For the big arena
                          # length = self.magnitudes[z] * 1 , arrow_length_ratio = .25, color = colors[i, j])
                          length = .00001, arrow_length_ratio=10000, color=colors[i, j])

    ax.set_xlabel('Longitude (X)')
    ax.set_ylabel('Latitude (Y)')
    ax.set_zlabel('Pressure Level (Z)')
    print(ds.longitude[0].values, ds.longitude[-1].values)
    #ax.set_xlim(50, 75)
    #ax.set_ylim(ds.latitude[0].values, ds.latitude[-1].values)
    #ax.set_ylim(0, (self.y_dim) / self.res)
    #ax.set_zlim(0, self.z_dim)  # Set the z-axis limit to the maximum altitude




    #Change ticks

    # Get current axis limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # Setting custom tick labels without changing the plot bounds
    z_ticks = ax.get_zticks()
    print(z_ticks)
    z_ticks_reversed = z_ticks[::-1]
    ax.set_zticks(z_ticks_reversed)
    ax.set_zlim(levels[-1], levels[0])
    plt.xticks(np.linspace(x_min, x_max, 5), np.linspace(ds.longitude[0].values, ds.longitude[-1].values, 5, dtype=int))
    plt.yticks(np.linspace(y_min, y_max, 5), np.linspace(ds.latitude[0].values, ds.latitude[-1].values, 5, dtype=int))

    '''
    # Customizing tick positions on the x-axis
    new_y_ticks = np.linspace(ds.latitude[-1].values, ds.latitude[0].values, 5)  # Example: ticks at 0, 2, 4, 6, 8, 10
    ax.set_yticks(new_y_ticks)

    new_x_ticks = np.linspace(ds.longitude[0].values, ds.longitude[-1].values, 5)  # Example: ticks at 0, 2, 4, 6, 8, 10
    ax.set_xticklabels(new_x_ticks)
    '''


fig = plt.figure(figsize=(15, 10))
skip = 6
res = 1

# Plot the flow field
ax1 = fig.add_subplot(111, projection='3d')
visualize_3d_planar_flow(ax1, skip, interpolation_point=None)

print(ds)

print(ds['u'].shape)

u_new = ds['u'].interp(time =ds['time'][0], latitude=38.0, longitude=-100.0, level=200).values
v_new = ds['v'].interp(time =ds['time'][0], latitude=38.0, longitude=-100.0, level=200).values

print(u_new, v_new)

plt.figure()


#let's visualize on a map...

#ds.drop_dims('level')

u_wind = ds['u'].sel(time = ds['time'][0], level=1 )
v_wind = ds['v'].sel(time = ds['time'][0], level=1 )

#sdfsdfs
u_wind.plot()
plt.figure()
v_wind.plot()

plt.figure()

ds.sel(time = ds['time'][0], level=1).plot.quiver(x="longitude", y="latitude", u="u", v="v")


#plt.show()



###################

# Coordinates and time index for plotting
lon = ds.longitude.values  # Longitude values
lat = ds.latitude.values   # Latitude values
level = ds.level.values    # Level values
time_index = 0             # Index of the time dimension you want to plot

# Extract u, v, w at the specified time index
u = ds.u.isel(time=time_index).values
v = ds.v.isel(time=time_index).values
w = np.zeros_like(u)

# Create a meshgrid of longitude, latitude, and level
lon_mesh, lat_mesh, level_mesh = np.meshgrid(lon, lat, level, indexing='ij')

# Ensure u, v, w have the same shape as lon_mesh, lat_mesh, level_mesh
# Transpose u, v, w to match the order (lon, lat, level)
u = np.transpose(u, (2, 1, 0))  # (C, B, A) -> (A, B, C)
v = np.transpose(v, (2, 1, 0))  # (C, B, A) -> (A, B, C)
w = np.transpose(w, (2, 1, 0))  # (C, B, A) -> (A, B, C)

# Create a 3D quiver plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

skip = (slice(None, None, 5), slice(None, None, 5), slice(None, None, 1))

# Plot the quiver
ax.quiver(lon_mesh[skip], lat_mesh[skip], level_mesh[skip], u[skip], v[skip], w[skip],
          length=.1,color='b', pivot='tail', arrow_length_ratio=0.001)


# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Level')
ax.set_title(f'3D Quiver Plot at Time Index {time_index}')

# Set limits based on data range
ax.set_xlim(lon.min(), lon.max())
ax.set_ylim(lat.min(), lat.max())
ax.set_zlim(level.min(), level.max())

plt.show()




