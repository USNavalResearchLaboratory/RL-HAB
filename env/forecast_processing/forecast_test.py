"""
Tests to make sure that the manual interpretation methods match with xarray, scipy, and numpy methods

Our manually written interpolation methods are faster for simulating
"""


import unittest
from env.forecast_processing.forecast import Forecast, Forecast_Subset
from env.config.env_config import env_params
from datetime import datetime

class TestFunctionOutputs(unittest.TestCase):
    def setUp(self):
        #Try some different variants here:
        #FORECAST_PRIMARY = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=7)
        #FORECAST_PRIMARY = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=7, timewarp=3)
        #FORECAST_PRIMARY = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=None, timewarp=3)
        FORECAST_PRIMARY = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH")
        FORECAST_PRIMARY = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH", timewarp=6)

        self.forecast_subset = Forecast_Subset(FORECAST_PRIMARY)
        self.forecast_subset.randomize_coord()
        print("random_coord", self.forecast_subset.lat_central, self.forecast_subset.lon_central, self.forecast_subset.start_time)
        self.forecast_subset.subset_forecast()

    def test1(self):
        lat = 35.22
        lon = -106.42
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test2(self):
        lat = 35.1
        lon = -105.57
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test3(self):
        lat = 35.17
        lon = -105.6
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    #Out of Bounds tests
    def test4(self):
        lat = 33
        lon = -106
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test5(self):
        lat = 38
        lon = -106
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test6(self):
        lat = 34
        lon = -101
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test7(self):
        lat = 34
        lon = -115
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test8(self):
        lat = 35.22
        lon = -106.42
        time = datetime.strptime('2022-08-15 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test9(self):
        lat = 35.22
        lon = -106.42
        time = datetime.strptime('2022-08-27 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")

    def test10(self):
        lat = 35.18
        lon = -106.43
        time = datetime.strptime('2022-08-22 15:33:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast_subset.xr_lookup(lat, lon, time)
        output2 = self.forecast_subset.np_lookup(lat, lon, time)
        self.assertEqual(output1[0].all(), output2[0].all(), "test")


if __name__ == "__main__":
    unittest.main()