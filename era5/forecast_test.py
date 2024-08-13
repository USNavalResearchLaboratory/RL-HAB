# test_functions.py
import unittest
from forecast import Forecast
from datetime import datetime, timedelta


class TestFunctionOutputs(unittest.TestCase):
    def setUp(self):
        pres_min = 20
        pres_max = 200
        rel_dist = 150000

        self.forecast = Forecast(rel_dist, pres_min, pres_max)


    def test1(self):
        lat = 35.22
        lon = -106.42
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test2(self):
        lat = 35.1
        lon = -105.57
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test3(self):
        lat = 35.17
        lon = -105.6
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    #Out of Bounds tests
    def test4(self):
        lat = 33
        lon = -106
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test5(self):
        lat = 38
        lon = -106
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test6(self):
        lat = 34
        lon = -101
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test7(self):
        lat = 34
        lon = -115
        time = datetime.strptime('2022-08-22 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test8(self):
        lat = 35.22
        lon = -106.42
        time = datetime.strptime('2022-08-15 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test9(self):
        lat = 35.22
        lon = -106.42
        time = datetime.strptime('2022-08-27 14:00:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")

    def test10(self):
        lat = 35.18
        lon = -106.43
        time = datetime.strptime('2022-08-22 15:33:00', "%Y-%m-%d %H:%M:%S")

        output1 = self.forecast.xr_lookup(lat,lon,time)
        output2 = self.forecast.np_lookup(lat,lon,time)
        self.assertEqual(output1.all(), output2.all(), "test")


if __name__ == "__main__":
    unittest.main()