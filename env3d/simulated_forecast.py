import xarray as xr
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

# Define the dimensions and coordinates
x = np.arange(10)
y = np.arange(10)
levels = [0,5,20,50,100]

# Create some data arrays for z, u, t
z = np.random.rand(len(levels), len(y), len(x))
u = np.random.rand(len(levels), len(y), len(x))
t = np.random.rand(len(levels), len(y), len(x))

# Create the xarray dataset
ds = xr.Dataset(
    {
        "z": (["level", "y", "x"], z),
        "u": (["level", "y", "x"], u),
        "v": (["level", "y", "x"], t)
    },
    coords={
        "x": x,
        "y": y,
        "level": levels
    }
)


# Add dataset attributes
ds.attrs["creation_datetime"] = datetime.now().isoformat()

print(ds)


def interpolate_xr(x,y,z):

    # Perform interpolation on x, y, and level dimensions
    interp_ds = ds.interp(x=x, y=y, level=ds['level'])

    # Now interpolate the result over z
    # Extract interpolated values and the corresponding z-values
    z_interp_values = interp_ds['z'].values
    u_interp_values = interp_ds['u'].values
    v_interp_values = interp_ds['v'].values

    # Create a function to perform interpolation over z
    def interpolate_over_z(z_values, data_values, target_z):
        print(z_values)
        print(data_values)
        print(target_z)
        return griddata(z_values, data_values, target_z, method='linear')

    # Interpolate the data variables over z
    interp_u = interpolate_over_z(z_interp_values, u_interp_values, z)
    interp_v = interpolate_over_z(z_interp_values, v_interp_values, z)

    # Create a new dataset with the interpolated values
    final_ds = xr.Dataset(
        {
            "u": ([], interp_u),
            "v": ([], interp_v)
        },
        coords={
            "x": x,
            "y": y,
            "z": z
        }
    )

    return final_ds

# Define new coordinates for interpolation
new_x = 5.5
new_y = 5.5
new_z = 0.5  # Example z-value to interpolate

final_ds = interpolate_xr(new_x,new_y,new_z)
print(final_ds)