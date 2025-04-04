import xarray as xr
from env.config.env_config import env_params
import pandas as pd
from termcolor import colored


filename = env_params["synth_netcdf"]
ds = xr.open_dataset(env_params["forecast_directory"] + filename)
'''
for var in ds.data_vars:
    for dim in ds[var].dims:
        try:
            ds[var] = ds[var].interpolate_na(dim=dim, method='linear')
        except Exception as e:
            print(f"Skipping {var} along {dim} due to error: {e}")
'''

'''
print(ds['time'].values)
is_monotonic = pd.Index(ds['time'].values).is_monotonic_increasing
print("Time monotonic?", is_monotonic)

time_vals = pd.Index(ds['time'].values)
print("Has duplicates?", time_vals.has_duplicates)
'''

nans_exist = False
print(ds)
for var in ds.data_vars:
    has_nan = ds[var].isnull().any()
    
    if has_nan.item():
        print(colored(f"WARNING: {var}: contains NaN? {has_nan.item()}", "yellow"))
    nans_exist = True

if nans_exist:
    for var in ds.data_vars:
        total_vals = ds[var].size
        n_nans = ds[var].isnull().sum().item()
        pct_nans = (n_nans / total_vals) * 100
        print(f"{var}: {n_nans} NaNs out of {total_vals} values ({pct_nans:.2f}%)")

    print(colored(f"Linear Filling Nans on time dimension", "yellow"))
    ds['u'] = ds['u'].interpolate_na(dim='time', method='linear', use_coordinate=False)
    ds['v'] = ds['v'].interpolate_na(dim='time', method='linear', use_coordinate=False)
    ds['z'] = ds['z'].interpolate_na(dim='time', method='linear', use_coordinate=False)
    print(colored(f"Linear Interpolation done", "yellow"))

print(ds)

#print(ds.isnull().any())

'''
print(ds.isnull().any().to_array().any().item())

for var in ds.data_vars:
    has_nan = ds[var].isnull().any()
    print(f"{var}: contains NaN? {has_nan.item()}")


nan_locations = ds['u'].isnull()
print(nan_locations.where(nan_locations, drop=True))

nan_locations = ds['v'].isnull()
print(nan_locations.where(nan_locations, drop=True))

nan_locations = ds['z'].isnull()
print(nan_locations.where(nan_locations, drop=True))'
'''