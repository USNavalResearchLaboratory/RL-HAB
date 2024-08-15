from era5 import config_earth
from era5 import ERA5
import xarray as xr
import numpy as np
from utils import CoordinateTransformations as transform
from env3d.config.env_config import env_params
from utils import constants, convert_range
from utils.convert_range import convert_range
from utils import CoordinateTransformations as transform
from line_profiler import profile
from datetime import datetime, timedelta

class Forecast:
    #Load from config file for now.  Maybe change this later
    def __init__(self, filename):

        self.load_forecast(filename)

        #self.randomize_coord()

    def quarter(self,x):
        return round(x*4)/4

    def load_forecast(self, filename):
        print("Loading Forecast...")
        self.ds_original = xr.open_dataset("forecasts/" + filename)

        # Reverse order of latitude, since era5 comes reversed for some reason?
        self.ds_original = self.ds_original.reindex(latitude=list(reversed(self.ds_original.latitude)))

        self.LAT_MIN = self.ds_original.latitude.values[0]
        self.LAT_MAX = self.ds_original.latitude.values[-1]
        self.LAT_DIM = len(self.ds_original.latitude.values)

        self.LON_MIN = self.ds_original.longitude.values[0]
        self.LON_MAX = self.ds_original.longitude.values[-1]
        self.LON_DIM = len(self.ds_original.longitude.values)

        self.LEVEL_MIN = self.ds_original.level.values[0]
        self.LEVEL_MAX = self.ds_original.level.values[-1]
        self.LEVEL_DIM = len(self.ds_original.level.values)

        #Assuming Forecasts are downloaded in 1 hour resolution for now?
        self.TIME_MIN = self.ds_original.time.values[0]
        self.TIME_MAX = self.ds_original.time.values[-1]
        self.TIME_DIM = len(self.ds_original.time.values)

        print(f"LAT RANGE: ({self.LAT_DIM }) {self.LAT_MIN}, {self.LAT_MAX}")
        print(f"LON RANGE: ({self.LON_DIM}) {self.LON_MIN}, {self.LON_MAX}")
        print(f"PRES RANGE: ({self.LEVEL_DIM}) {self.LEVEL_MIN}, {self.LEVEL_MAX}")
        print(f"TIME RANGE: ({self.TIME_DIM}) {self.TIME_MIN}, {self.TIME_MAX}")

        print(self.LAT_MIN, self.LAT_MAX)

        print("Forecast Loaded")

        #return lat, lon, time

class Forecast_Subset:
    # Load from config file for now.  Maybe change this later
    def __init__(self, Forecast):
        self.Forecast = Forecast

    def quarter(self,x):
        return round(x*4)/4

    def randomize_coord(self):
        lat = np.random.uniform(low=self.Forecast.LAT_MIN+2, high=self.Forecast.LAT_MAX-2)
        lon = np.random.uniform(low=self.Forecast.LON_MIN + 2, high=self.Forecast.LON_MAX - 2)
        #Convert time to unix for randomizing.
        # Subtract 24 hours from the end for simulating.
        time = np.random.uniform(low=self.get_unixtime(self.Forecast.TIME_MIN), high=self.get_unixtime(self.Forecast.TIME_MAX-np.timedelta64(23, "h")))
        # Convert time back to dt64
        time = np.datetime64(int(time),'s')

        #Round time to nearest hour and quarter
        self.start_time = np.array(time, dtype='datetime64[h]')
        self.lat_central = self.quarter(lat)
        self.lon_central = self.quarter(lon)


    def subset_forecast(self, rel_dist, pres_min, pres_max):


        #1.  Calculate Lat/Lon Coordinates for subsetting the data to The relative distance area
        lat_min, _ = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, 0, -rel_dist)
        _, lon_min = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, -rel_dist, 0)

        lat_max, _ = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, 0, rel_dist)
        _ , lon_max = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, rel_dist, 0)

        #Round to nearest .25 degree resolution since that's the res of ERA5 forecasts
        self.lat_min = self.quarter(lat_min)
        self.lon_min = self.quarter(lon_min)

        self.lat_max = self.quarter(lat_max)
        self.lon_max = self.quarter(lon_max)

        #2. Subset the forecast to a smaller array #This may not be necessary for simulating, but good for forecast visualization)
        #No time for now?
        self.ds = self.Forecast.ds_original.sel(latitude=slice(self.lat_min, self.lat_max),
                              longitude=slice(self.lon_min, self.lon_max),
                              level=slice(pres_min,pres_max),
                              time=slice(self.start_time, self.start_time + np.timedelta64(24, "h"))) #1 day of time for now


        # 3. Determine new min and max values from the subsetted forecast

        self.lat_min = self.ds.latitude.values[0]
        self.lat_max = self.ds.latitude.values[-1]

        self.lon_min = self.ds.longitude.values[0]
        self.lon_max = self.ds.longitude.values[-1]

        self.start_time = self.ds.time.values[0]
        self.end_time = self.ds.time.values[-1]


        #Now Calculate new Dimensions
        self.lat_dim = len(self.ds.latitude)
        self.lon_dim = len(self.ds.longitude)
        self.level_dim = len(self.ds.level)
        self.time_dim = len(self.ds.time)

        # Convert the subset forecast Dataset to a numpy array for faster processing
        self.forecast_np = self.ds.to_array()
        self.forecast_np = self.forecast_np.to_numpy()
        self.pressure_levels = self.ds.level.values

    @profile
    def xr_lookup(self,lat, lon, timestamp):
        # How to look up indicies for xarray
        #time_idx = list(self.ds.time.values).index(self.ds.sel(time=timestamp, method='nearest').time)
        #lat_idx = list(self.ds.latitude.values).index(self.ds.sel(latitude=lat, method='nearest').latitude)
        #lon_idx = list(self.ds.longitude.values).index(self.ds.sel(longitude=lon, method='nearest').longitude)

        return self.ds.sel(latitude=lat, longitude=lon,  time=timestamp, method = "nearest")['z'].values

    def get_unixtime(self, dt64):
        return dt64.astype('datetime64[s]').astype('int')


    @profile
    def np_lookup(self,lat, lon, time):
        lat_idx = int(convert_range(lat, self.lat_min,self.lat_max, 0, self.lat_dim))
        lon_idx = int(convert_range(lon, self.lon_min, self.lon_max, 0, self.lon_dim))
        time_idx = int(convert_range(self.get_unixtime(np.datetime64(time)), self.get_unixtime(self.start_time), self.get_unixtime(self.end_time), 0, self.time_dim))

        # Clip idx's to out of bounds.  Should I add a warning here?
        lat_idx = np.clip(lat_idx, 0, self.lat_dim-1)
        lon_idx = np.clip(lon_idx, 0, self.lon_dim - 1)
        time_idx = np.clip(time_idx, 0, self.time_dim - 1)

        z = self.forecast_np[0, time_idx, :, lat_idx, lon_idx] / constants.GRAVITY
        u = self.forecast_np[2, time_idx, :, lat_idx, lon_idx]
        v = self.forecast_np[3, time_idx, :, lat_idx, lon_idx]

        return (z,u,v)

    def windVectorToBearing(self, u, v):
        """Helper function to conver u-v wind components to bearing and speed.

        """
        bearing = np.arctan2(v,u)
        speed = np.power((np.power(u,2)+np.power(v,2)),.5)

        return [bearing, speed]

    def interpolate_wind(self, alt, z,u,v):
        #Ignoring time for right now

        #need to change to ascending order for interpolating with numpy
        u_wind = np.interp(alt, z[::-1], u[::-1])
        v_wind = np.interp(alt, z[::-1], v[::-1])

        #bearing, speed = self.windVectorToBearing(u_wind, v_wind)

        #Do I need to do the axis crossover check now if not interpolating 2 times?

        return u_wind, v_wind

    def getNewCoord(self, Balloon, SimulationState, dt):
        #Balloon lat and lon coordinates should dictate relative X and Y coordinates.  I think....


        #get Wind at current lat/lon
        z_col, u_col, v_col = self.np_lookup(Balloon.lat, Balloon.lon, SimulationState.timestamp)
        x_vel, y_vel = self.interpolate_wind(Balloon.altitude, z_col, u_col, v_col )

        #Get current relative X,Y Position
        relative_x, relative_y = transform.latlon_to_meters_spherical(self.lat_central,
                                                                self.lon_central,
                                                                Balloon.lat, Balloon.lon)


        #Apply the velocity to relative Position
        x_new = relative_x + x_vel * dt
        y_new =  relative_y + y_vel * dt

        # Convert New Relative Position back to Lat/Lon
        lat_new, lon_new = transform.meters_to_latlon_spherical(self.lat_central,
                                                                self.lon_central,
                                                                x_new, y_new)


        return [lat_new, lon_new, x_vel, y_vel, x_new, y_new ]


if __name__ == '__main__':
    pres_min = env_params['pres_min']
    pres_max = env_params['pres_max']
    rel_dist = env_params['rel_dist']

    filename = "SHAB14V_ERA5_20220822_20220823.nc"
    FORECAST_PRIMARY = Forecast(filename)

    forecast_subset = Forecast_Subset(FORECAST_PRIMARY)
    forecast_subset.randomize_coord()
    print("random_coord", forecast_subset.lat_central, forecast_subset.lon_central, forecast_subset.start_time)
    forecast_subset.subset_forecast(rel_dist, pres_min, pres_max)

