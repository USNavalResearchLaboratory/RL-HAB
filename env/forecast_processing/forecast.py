"""
Current Assumptions for Synth and ERA5 forecast

- Both forecasts are downloaded for the same regions (0.25 degree resolution), same altitude band (config)
- Synth Forecast is 1 month @ 12 hour intervals
- Synth Forecast is 6 months @ 3 hour intervals

"""

import xarray as xr
import numpy as np
from env.config.env_config import env_params
from utils import constants
from utils.common import convert_range, quarter
from utils import CoordinateTransformations as transform
from line_profiler import profile
from termcolor import colored
import pandas as pd

class Forecast:
    """
        Loads a full ERA5 or synthetic forecast into memory.

        This class handles large-scale climate forecasts used for simulations, supporting operations
        like subsetting, time adjustments, and aligning forecasts for simulating.

        Download from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

        Attributes:
            forecast_type (str): Type of forecast ('SYNTH' or 'ERA5').
            ds_original (xarray.Dataset): Original dataset loaded from the forecast file.
            LAT_MIN, LAT_MAX (float): Latitude range of the forecast.
            LON_MIN, LON_MAX (float): Longitude range of the forecast.
            LEVEL_MIN, LEVEL_MAX (float): Pressure level range of the forecast.
            TIME_MIN, TIME_MAX (numpy.datetime64): Time range of the forecast.
        """
    def __init__(self, filename, forecast_type = None, month = None, timewarp=None ):
        """
        Initialize the Forecast object and load a dataset.

        Args:
            filename (str): Path to the forecast file.
            forecast_type (str): Type of forecast ('SYNTH' or 'ERA5').
            month (int, optional): Month to filter for ERA5 forecasts.
            timewarp (int, optional): Simulation Time interval adjustment (e.g., 3, 6, or 12 hours).

        Raises:
            Exception: If the forecast type is invalid.
        """

        self.forecast_type = forecast_type

        self.load_forecast(filename, month, timewarp=timewarp)

        #check and see if the forecast type is correct
        if forecast_type != "SYNTH" and forecast_type != "ERA5":
            raise Exception("Invalid forecast type " + str(forecast_type))


    def load_forecast(self, filename, month = None, timewarp = None):
        """
        Load and preprocess the forecast dataset.

        Args:
            filename (str): Path to the forecast file.
            month (int, optional): Month to filter for ERA5 forecasts.
            timewarp (int, optional): Time interval adjustment (e.g., 3, 6, or 12 hours).
        """
        self.ds_original = xr.open_dataset(env_params["forecast_directory"] + filename)

        # Drop temperature variable from forecasts if it exists
        if 't' in self.ds_original.data_vars:
            self.ds_original = self.ds_original.drop_vars('t')

        # Do some reformatting for the new format of ERA5
        if 'valid_time' in self.ds_original.coords:
            #self.ds_original = self.ds_original.drop_vars('expver')
            #self.ds_original = self.ds_original.drop_vars('number')
            self.ds_original = self.ds_original.rename({'valid_time': 'time','pressure_level': 'level'})

            #self.ds_original['latitude'] = self.ds_original['latitude'].astype('float32')
            #self.ds_original['longitude'] = self.ds_original['longitude'].astype('float32')

            #reformat the pressure level to match the old format
            #self.ds_original['level'] = self.ds_original['level'].astype('int32')
            #Reverse the 'level' coordinate
            self.ds_original = self.ds_original.reindex(level=self.ds_original.level[::-1])
            print("DID WE GO IN HERE")

        if self.forecast_type == "ERA5":
            memory_size_gb = self.ds_original.nbytes / 1e9
            # Print the memory size of the dataset in gigabytes
            print(f"Memory size of the dataset: {memory_size_gb:.6f} GB")

        # Reverse order of latitude, since era5 comes reversed for some reason (We set up synth to be the same)
        self.ds_original = self.ds_original.reindex(latitude=list(reversed(self.ds_original.latitude)))


        # Cut off any pressure levels that are not in range of the altitude (only checking top bounds right now)
        da_slice = self.ds_original.isel(time=0, latitude=0, longitude=0)
        idx = np.argmin(np.abs(da_slice.z.values / 9.81 - env_params["alt_max"]))
        max_pres_level = da_slice['level'][idx].level.values
        self.ds_original = self.ds_original.sel(level=slice(max_pres_level, None))
        #print('Max_pres level', max_pres_level)


        # Some forecast formatting helper functions that are performed by default with v1.0
        # Only include same ERA5 month as Synth, unless month is not specified
        # Need to format ERA5 before timewarping
        if self.forecast_type == "ERA5" and month != None:
            self.drop_era5_months(month)


        # Change the simulation timestamps of forecasts
        if timewarp != None:
            self.TIMEWARP(timewarp)

        print(colored(self.forecast_type, "green"))
        print(self.ds_original)

        # Set master forecast variables
        self.LAT_MIN = self.ds_original.latitude.values[0]
        self.LAT_MAX = self.ds_original.latitude.values[-1]
        self.LAT_DIM = len(self.ds_original.latitude.values)

        self.LON_MIN = self.ds_original.longitude.values[0]
        self.LON_MAX = self.ds_original.longitude.values[-1]
        self.LON_DIM = len(self.ds_original.longitude.values)

        self.LEVEL_MIN = self.ds_original.level.values[0]
        self.LEVEL_MAX = self.ds_original.level.values[-1]
        self.LEVEL_DIM = len(self.ds_original.level.values)

        self.TIME_MIN = self.ds_original.time.values[0]
        self.TIME_MAX = self.ds_original.time.values[-1]
        self.TIME_DIM = len(self.ds_original.time.values)

        #print(f"LAT RANGE: ({self.LAT_DIM }) {self.LAT_MIN}, {self.LAT_MAX}")
        #print(f"LON RANGE: ({self.LON_DIM}) {self.LON_MIN}, {self.LON_MAX}")
        #print(f"PRES RANGE: ({self.LEVEL_DIM}) {self.LEVEL_MIN}, {self.LEVEL_MAX}")
        #print(f"TIME RANGE: ({self.TIME_DIM}) {self.TIME_MIN}, {self.TIME_MAX}")


    def drop_era5_months(self, month):
        """
        Filter ERA5 forecast to match the specified month and reduce time intervals to 12 hours. This is a bit hardcoded rn

        By default, ERA5 forecasts are downloaded in 6 month @ 3 hour intervals. Since many simulations require
        both Synth and ERA5, we need change ERA5 to 12 hour intervals

        Args:
            month (int): Month to retain in the dataset.

        Raises:
            Exception: If the specified month is not within the forecast's range.
        """

        print(colored("DROPPING ERA5 Months except (" + str(month) + ") and  Times: (12) hour intervals", "yellow"))

        # for error printing
        start_time = self.ds_original.time.values[0]
        end_time = self.ds_original.time.values[-1]

        # Reformat the ERA5 forecast to only have times every 12 hours like Synth Forecasts
        # Also Only include the same month if ERA5 has more than a month
        month_mask = self.ds_original.time.dt.month == month
        hour_mask = self.ds_original.time.dt.hour.isin([0, 12])
        combined_mask = month_mask & hour_mask
        self.ds_original = self.ds_original.sel(time=combined_mask)

        if self.ds_original.time.size == 0:
            raise Exception(
                f"Month {month} is out of range of ERA forecast with time range of {start_time} - {end_time}")


    def TIMEWARP(self, timewarp):
        '''
        By Default ERA5 forecasts are downloaded in 3 hour intervals, whereas Synth are in 12 hour intervals

        Therefore we perform a "timewarp" to overwrite timestamps in both forecasts, while still matching
        up the data from the original timestamps with the new timestamps.

        For example:

        Synth (original):   2024-01-01 00:00:00, 2024-01-01 12:00:00, 2024-01-02 00:00:00, 2024-01-02 12:00:00
        *original timestamps will be overwritten with the timewarp function
        Synth (timewarp) :  2024-01-01 00:00:00, 2024-01-01 03:00:00, 2024-01-01 06:00:00, 2024-01-01 09:00:00
        *to line up with the ERA5
        ERA5:               2024-01-01 00:00:00, 2024-01-01 03:00:00, 2024-01-01 06:00:00, 2024-01-01 09:00:00

        Timewarping will typically only be used for Synth Forecast due to their sparse timing
        '''

        if timewarp != 1 and timewarp != 3 and timewarp != 6 and timewarp != 12:
            raise Exception(colored("Timewarp only accepts hour intervals of 1,3,6,12", "yellow"))

        print(colored("TIMEWARPING (" + self.forecast_type + ")", "cyan"))

        # determine initial time variables (temporary, not assigned)
        time_min = self.ds_original.time.values[0]
        time_dim = len(self.ds_original.time.values)

        # Add timewarp time interval to the forecast
        synth_simulated_time = []
        for i in range(0, time_dim):
            synth_simulated_time.append(time_min + np.timedelta64(i * timewarp, "h"))

        self.ds_original['time'] = synth_simulated_time
        self.ds_original = self.ds_original.reindex(time=synth_simulated_time)

        #print(self.ds_original.time)
        #sdfsdf


class Forecast_Subset:
    """
    Creates a subset of the master forecast for efficient processing and simulation.

    Attributes:
        Forecast (Forecast): Master forecast object.
        lat_central (float): Central latitude of the subset.
        lon_central (float): Central longitude of the subset.
        start_time (numpy.datetime64): Start time of the subset.
        ds (xarray.Dataset): Subset dataset.
    """
    # Load from config file for now.  Maybe change this later
    def __init__(self, Forecast):
        """
        Initialize the Forecast_Subset object.

        Args:
            Forecast (Forecast): Master forecast object.
        """
        self.Forecast = Forecast

    def assign_coord(self, lat, lon, timestamp):
        """
        Assign central coordinates and timestamp for the subset.

        Args:
            lat (float): Central latitude.
            lon (float): Central longitude.
            timestamp (numpy.datetime64): Start timestamp.
        """
        # Round time to nearest hour and quarter
        self.start_time = np.array(timestamp, dtype='datetime64[h]')
        self.lat_central = quarter(lat)
        self.lon_central = quarter(lon)

    def randomize_coord(self, np_rng):
        """
        Generates a random coordinate to centralize the Forecast Subset, and stores the coordinate for look up by other classes.

        Altitude Bounds are the same as the PRIMARY FORECAST
        Horizontal Bounds are within 2 degrees of the min/max LAT/LON from the PRIMARY FORECAST
        Time Bounds are between the start time and up to 24 hours before the final timestamp of the PRIMARY FORECAST

        pass np_rng to have forecasts randomize in the same order when manually setting seed

        Args:
            np_rng (numpy.random.Generator): Random number generator.
        """

        #print("RANDOM NUMBER", np_rng.uniform(low=0, high=100))
        #print("RANDOM NUMBER 2", np.random.uniform(low=0, high=100))

        lat = np_rng.uniform(low=self.Forecast.LAT_MIN+2, high=self.Forecast.LAT_MAX-2)
        lon = np_rng.uniform(low=self.Forecast.LON_MIN + 2, high=self.Forecast.LON_MAX - 2)
        #Convert time to unix for randomizing.
        # Subtract 24 hours from the end for simulating.
        time = np_rng.uniform(low=self.get_unixtime(self.Forecast.TIME_MIN), high=self.get_unixtime(self.Forecast.TIME_MAX-np.timedelta64(24, "h")))
        # Convert time back to dt64
        time = np.datetime64(int(time),'s')

        #Round time to nearest hour and quarter
        self.start_time = np.array(time, dtype='datetime64[h]')
        self.lat_central = quarter(lat)
        self.lon_central = quarter(lon)

        self.fourecast_error_count = 0


    def get_alt_from_pressure(self, pressure):
        """Get average altitude from ERA5 for a forecast subset. Average is taken since z is geopotential converted
        to altitude

        Args:
            pressure (float): atmospheric pressure.

        Returns:
            alt: corresponding altitude (from geopotential) for pressure level
        """
        try:
            # Use sel() to find the matching index
            alt_array = self.ds.sel({"level": pressure}).z.values/9.81
            avg_alt = np.mean(alt_array)


            return avg_alt
        except KeyError:
            print(colored(f"Value {pressure} doesn't exist in the levels.","yellow"))
            # Return a message if the value is not found
            return None


    def subset_forecast(self, days = 1):
        """
        Subsets the Forecast to the central coordinate. This assume a random coordinate or user input coordinate has already been assigned.

        Horizontal Bounds are determined by the relative distance (converted to lat/lon degrees)
        Altitude is the same
        Time is 24 hours

        Converts the DataSet to a numpy array for faster processing

        Args:
            days (int): Number of days to include in the subset.

        """

        rel_dist = env_params['rel_dist']
        pres_min = env_params['pres_min']
        pres_max = env_params['pres_max']

        #1.  Calculate Lat/Lon Coordinates for subsetting the data to The relative distance area
        lat_min, _ = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, 0, -rel_dist)
        _, lon_min = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, -rel_dist, 0)

        lat_max, _ = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, 0, rel_dist)
        _ , lon_max = transform.meters_to_latlon_spherical(self.lat_central, self.lon_central, rel_dist, 0)

        #Round to nearest .25 degree resolution since that's the res of ERA5 forecasts
        self.lat_min = quarter(lat_min)
        self.lon_min = quarter(lon_min)

        self.lat_max = quarter(lat_max)
        self.lon_max = quarter(lon_max)

        #print("SUBSETTING")


        #2. Subset the forecast to a smaller array
        # This may not be necessary for simulating, but good for forecast visualization)
        # No time for now?
        self.ds = self.Forecast.ds_original.sel(latitude=slice(self.lat_min, self.lat_max),
                              longitude=slice(self.lon_min, self.lon_max),
                              level=slice(pres_min,pres_max),
                              time=slice(self.start_time, self.start_time + np.timedelta64(days, "D"))) #1 day of time for now

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

        # Make sure the forecast subset has been changed to the right order:
        ordered_vars = ['z', 'u', 'v']# Reorder ds2 to match ds1
        self.ds = self.ds[ordered_vars]

        self.forecast_np = self.ds.to_array()
        self.forecast_np = self.forecast_np.to_numpy()
        self.pressure_levels = self.ds.level.values



    @profile
    def xr_lookup(self,lat, lon, timestamp):

        # How to look up indicies for xarray
        #time_idx = list(self.ds.time.values).index(self.ds.sel(time=timestamp, method='nearest').time)
        #lat_idx = list(self.ds.latitude.values).index(self.ds.sel(latitude=lat, method='nearest').latitude)
        #lon_idx = list(self.ds.longitude.values).index(self.ds.sel(longitude=lon, method='nearest').longitude)

        z = self.ds.sel(latitude=lat, longitude=lon, time=timestamp, method="nearest")['z'].values / constants.GRAVITY
        u = self.ds.sel(latitude=lat, longitude=lon, time=timestamp, method="nearest")['u'].values
        v = self.ds.sel(latitude=lat, longitude=lon, time=timestamp, method="nearest")['v'].values

        return (z,u,v)

    def get_unixtime(self, dt64):
        """
        Convert numpy.datetime64 to Unix time in seconds.

        Args:
            dt64 (numpy.datetime64): DateTime value.

        Returns:
            int: Unix timestamp in seconds.
        """
        return dt64.astype('datetime64[s]').astype('int')


    @profile
    def np_lookup(self,lat, lon, time):
        """
        Perform a fast lookup for wind data using numpy arrays.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            time (numpy.datetime64): Time.

        Returns:
            tuple: Altitude, u-component, and v-component of the wind.
        """
        lat_idx = int(convert_range(lat, self.lat_min, self.lat_max, 0, self.lat_dim))
        lon_idx = int(convert_range(lon, self.lon_min, self.lon_max, 0, self.lon_dim))
        time_idx = int(convert_range(self.get_unixtime(np.datetime64(time)), self.get_unixtime(self.start_time),
                                     self.get_unixtime(self.end_time), 0, self.time_dim))

        # Cast again see if that fixes the problem
        # Clip idx's to out of bounds.  Should I add a warning here?
        lat_idx = int(np.clip(lat_idx, 0, self.lat_dim - 1))
        lon_idx = int(np.clip(lon_idx, 0, self.lon_dim - 1))
        time_idx = int(np.clip(time_idx, 0, self.time_dim - 1))


        z = self.forecast_np[0, time_idx, :, lat_idx, lon_idx] / constants.GRAVITY
        u = self.forecast_np[1, time_idx, :, lat_idx, lon_idx]
        v = self.forecast_np[2, time_idx, :, lat_idx, lon_idx]

        return (z,u,v)

    def windVectorToBearing(self, u, v):
        """Helper function to convert u-v wind components to bearing and speed.

        Not being used right now.

        """
        bearing = np.arctan2(v,u)
        speed = np.power((np.power(u,2)+np.power(v,2)),.5)

        return bearing, speed

    def interpolate_wind(self, alt, z,u,v):
        '''
        Interpolates the u and v wind components given a 3D coordinate (lat,lon,alt)
        Currently only interpolating in the Z direction.  No smoothing for time or horizontal changes.
        '''

        #need to change to ascending order for interpolating with numpy
        u_wind = np.interp(alt, z[::-1], u[::-1])
        v_wind = np.interp(alt, z[::-1], v[::-1])

        return u_wind, v_wind

    def getNewCoord(self, Balloon, SimulationState, dt):
        """
        Determines new coordinate based on the flow at the current position and integrates forward int time via dt

        """

        #get Wind at current lat/lon
        z_col, u_col, v_col = self.np_lookup(Balloon.lat, Balloon.lon, SimulationState.timestamp)
        x_vel, y_vel = self.interpolate_wind(Balloon.altitude, z_col, u_col, v_col )

        #print("alt", Balloon.altitude, z_col, "u_vel:", u_col, "v_vel", v_col)


        #Get current relative X,Y Position
        relative_x, relative_y = transform.latlon_to_meters_spherical(self.lat_central,
                                                                self.lon_central,
                                                                Balloon.lat, Balloon.lon)

        #print()
        #print("alt", Balloon.altitude, "x_vel:", x_vel, "y_vel", y_vel)
        #print("relative_x", relative_x, "relative_y" , relative_y )

        #Apply the velocity to relative Position
        x_new = relative_x + x_vel * dt
        y_new =  relative_y + y_vel * dt



        # Convert New Relative Position back to Lat/Lon
        lat_new, lon_new = transform.meters_to_latlon_spherical(self.lat_central,
                                                                self.lon_central,
                                                                x_new, y_new)
        #print("x_new", x_new, "y_new", y_new)
        #print("pre_lat", Balloon.lat, "pre_lon", Balloon.lon)
        #print("lat_new", lat_new, "lon_new", lon_new)


        return [lat_new, lon_new, x_vel, y_vel, x_new, y_new ]


if __name__ == '__main__':
    #Can't use utils.initialize_forecast here, because it would be a circular import
    FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH", timewarp=3)
    # Get month associated with Synth
    synth_month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
    # Then process ERA5 to span the same timespan as a monthly Synthwinds File
    FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=synth_month, timewarp=3)


    forecast_subset = Forecast_Subset(FORECAST_ERA5) #Choose FORECAST_SYNTH or FORECAST_ERA5 here
    forecast_subset.randomize_coord()
    print("random_coord", forecast_subset.lat_central, forecast_subset.lon_central, forecast_subset.start_time)
    forecast_subset.subset_forecast()

    print(forecast_subset.ds)