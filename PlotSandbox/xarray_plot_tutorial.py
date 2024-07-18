import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import xarray as xr

airtemps = xr.tutorial.open_dataset("air_temperature")


air = airtemps.air - 273.15

print(airtemps)

print(air)

air2d = air.isel(time=500)

print(air2d)
air2d.plot()
plt.show()