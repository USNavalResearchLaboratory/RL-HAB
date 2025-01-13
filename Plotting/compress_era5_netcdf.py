
# New ERA5 format is compressed differently and used different data types than the old format (pre Setp 2024).

#From command line run:
'''
nccopy -d1 -c valid_time/100,pressure_level/10,latitude/50,longitude/50 ERA5-H2-2023-SEA.nc optimized_ERA5-H2-2023-SEA.nc
'''

#-d1: Applies light compression (adjust for balance between speed and size).
#-c: Defines chunk sizes optimized for your analysis patterns.

#Next we will change the types:

import xarray as xr

# Open the slow file
ds = xr.open_dataset('/mnt/d/FORECASTS/ERA5-2020-WH.nc')

if 'expver' in ds.data_vars:
    ds = ds.drop_vars('expver')
    ds = ds.drop_vars('number')

'''
# Re-encode variables
encoding = {}
for var in ['z', 'u', 'v']:
    encoding[var] = {'dtype': 'int16', 'scale_factor': ds[var].std().values / 32767,
                     'add_offset': ds[var].mean().values, '_FillValue': -32767}
'''

ds = ds.rename({'valid_time': 'time', 'pressure_level': 'level'})
ds = ds.reindex(level=ds.level[::-1])

# Save to a new file
#ds.to_netcdf('FORECASTS/optimized_ERA5-H2-2023-USA-updated.nc', encoding=encoding)
ds.to_netcdf('/mnt/d/FORECASTS/optimized_ERA5-2020-WH.nc')

#ncks -6 /mnt/d/FORECASTS/optimized_ERA5-2020-WH.nc /mnt/d/FORECASTS/optimized_ERA5-2020-WH.nc

print(ds)
