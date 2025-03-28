from netCDF4 import Dataset

file_path = "/mnt/d/FORECASTS/2023-ERA5-NORTH_renamed-condensed_64bit-2.nc"
filepath = "FORECASTS/ERA5-H1-2023-USA.nc"

try:
    ds = Dataset(file_path, "r")
    print("File format:", ds.file_format)
    print("Variables:", ds.variables.keys())

    if "z" in ds.variables:
        print("Attributes of z:", ds.variables["z"].ncattrs())
        print("z shape:", ds.variables["z"].shape)

        # Try reading z
        z_data = ds.variables["z"][:]
        print("z data loaded successfully.")

    ds.close()
except Exception as e:
    print(f"Error loading file: {e}")

import xarray as xr


ds = xr.open_dataset(file_path, decode_times=False)  # Disable datetime decoding
print(ds)  # Look at the variable names
#print(ds["v"])  # Inspect the time variable
#print(ds["v"].values)  # Check raw numeric values
