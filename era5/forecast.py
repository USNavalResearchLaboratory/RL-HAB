from era5 import config_earth
from era5 import ERA5
import xarray as xr
from utils import CoordinateTransformations as transform
from env3d.config.env_config import env_params

class Forecast:
    #Load from config file for now.  Maybe change this later
    def __init__(self, rel_dist, pres_min, pres_max):

        self.rel_dist = env_params['rel_dist'] # m relative distance from central coordiante

        # ERA5 stuff
        self.start_coord = config_earth.simulation['start_coord']

        self.load_forecast(rel_dist, pres_min, pres_max)

    def quarter(self,x):
        return round(x*4)/4

    def load_forecast(self, rel_dist, pres_min, pres_max):

        #load Forecast
        #Manual Upload for now?
        self.ds = xr.open_dataset("forecasts/" + config_earth.netcdf_era5['filename'])


        #Reverse order of latitude, since era5 comes reversed for some reason?
        self.ds = self.ds.reindex(latitude=list(reversed(self.ds.latitude)))

        self.start_coord = config_earth.simulation['start_coord']

        lat_min, _ = transform.meters_to_latlon_spherical(self.start_coord["lat"], self.start_coord["lon"], 0, -rel_dist)
        _, lon_min = transform.meters_to_latlon_spherical(self.start_coord["lat"], self.start_coord["lon"], -rel_dist, 0)

        lat_max, _ = transform.meters_to_latlon_spherical(self.start_coord["lat"], self.start_coord["lon"], 0, rel_dist)
        _ , lon_max = transform.meters_to_latlon_spherical(self.start_coord["lat"], self.start_coord["lon"], rel_dist, 0)

        #Round to nearest .25 degree resolution since that's the res of ERA5 forecasts
        self.lat_min = self.quarter(lat_min)
        self.lon_min = self.quarter(lon_min)

        self.lat_max = self.quarter(lat_max)
        self.lon_max = self.quarter(lon_max)

        #Do some subsetting of the data
        self.ds = self.ds.sel(latitude=slice(self.lat_min, self.lat_max), longitude=slice(self.lon_min, self.lon_max), level=slice(pres_min,pres_max))

        self.pressure_levels = self.ds.level.values

if __name__ == '__main__':
    pres_min = env_params['pres_min']
    pres_max = env_params['pres_max']
    rel_dist = env_params['rel_dist']

    gfs = ERA5.ERA5(config_earth.simulation['start_coord'])
    forecast = Forecast(rel_dist, pres_min, pres_max)

    print(forecast.ds)
