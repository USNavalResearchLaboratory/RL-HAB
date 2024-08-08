
from era5 import config_earth
import pandas as pd

class BalloonState(object):
    def __init__(self, x=None, y=None, z=None, x_vel=0, y_vel=0, z_vel=0, distance=None, lat=None, lon=None, rel_bearing=None):
        self.x = x
        self.y = y
        self.z = z

        self.x_vel = x_vel
        self.y_vel = y_vel
        self.z_vel = z_vel

        self.lat= lat
        self.lon= lon

        self.distance= distance
        self.rel_bearing = rel_bearing

        self.last_action= None

        self.wind_column= None

        #For later
        power= None

    def __str__(self):
        return (f"BalloonState(x={self.x}, y={self.y}, z={self.z},\n "
                f"dist={self.distance}, rel_bearing={self.rel_bearing},\n "
                f"x_vel={self.x_vel}, y_vel={self.y_vel}, z_vel={self.z_vel},\n "
                f"lat={self.lat}, lon={self.lon}),\n"
                f"last_action={self.last_action}, wind_column={self.wind_column})")

    def update(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)

class SimulatorState(object):
    def __init__(self,Balloon):

        self.Balloon = Balloon

        self.timestamp = config_earth.simulation['start_time']
        self.dt = config_earth.simulation['dt']

        self.trajectory = []
        self.time_history = []

    def step(self,Balloon):
        self.timestamp = self.timestamp + pd.Timedelta(hours=(1 / 3600 * self.dt))

        self.trajectory.append((Balloon.x , Balloon.y, Balloon.z))
        self.time_history.append(self.timestamp)

