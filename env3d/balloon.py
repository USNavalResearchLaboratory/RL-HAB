
from era5 import config_earth
import pandas as pd
import enum
from env3d.config.env_config import env_params

class BalloonState(object):
    '''Balloon State object for for updating and referencing during a simulation

    Attributes:
        x: Balloons relative x position in m converted from lat/lng to cartesian using a spherical coordinate
        transformation (see latlon_to_meters_spherical in utils)
        y: Balloons relative x position in m converted from lat/lng to cartesian using a spherical coordinate
        transformation (see latlon_to_meters_spherical in utils)

        altitude: Balloon's absolute (m)

        x_vel: Balloon's current velocity in m/s
        y_vel: Balloon's current velocity in m/s
        z_vel: Balloon's current ascent/descent velocity in m/s

        lat: The current latitude of balloon  (what numbering style?)
        lon: The current longitude of balloon (what numbering style?)

        distance: Current Distance to Station (XY plane) in m
        rel_bearing:  The relative bearing of the direction of motion the balloon is traveling to in reference to the station

        atm_pressure: The atmospheric pressure around the balloon at the current altitude
    '''


    def __init__(self, x=None, y=None, altitude=None, x_vel=0, y_vel=0, z_vel=0, distance=None, lat=None, lon=None, rel_bearing=None, pressure=None):
        self.x = x
        self.y = y
        self.altitude = altitude

        self.x_vel = x_vel
        self.y_vel = y_vel
        self.z_vel = z_vel

        self.lat= lat
        self.lon= lon

        self.distance= distance
        self.rel_bearing = rel_bearing

        self.atm_pressure = None  #Figure this out later

        self.last_action= None

        self.wind_column= None

        #For later
        power= None

    def __str__(self):
        return (f"BalloonState(x={self.x}, y={self.y}, altitude={self.altitude},\n "
                f"dist={self.distance}, rel_bearing={self.rel_bearing},\n "
                f"x_vel={self.x_vel}, y_vel={self.y_vel}, z_vel={self.z_vel},\n "
                f"lat={self.lat}, lon={self.lon}),\n"
                f"last_action={self.last_action}, wind_column={self.wind_column})")

    def update(self, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)


class AltitudeControlCommand(enum.IntEnum):
    """Altitude Control Command"""
    DOWN = 0
    STAY = 1
    UP = 2

class SimulatorState(object):
    '''
    A Simulator Class for keeping track of overall simulator state variables

    '''
    def __init__(self,Balloon):

        self.Balloon = Balloon

        self.timestamp = config_earth.simulation['start_time']
        self.dt = config_earth.simulation['dt']
        self.total_steps = 0

        self.episode_length = env_params['episode_length']

        self.trajectory = []
        self.time_history = []

    def step(self,Balloon):
        '''
        Steps the simulator global variables and updates accordingly.  This should match the step rate in gym environments

        :param Balloon:
        :return:
        '''

        self.timestamp = self.timestamp + pd.Timedelta(hours=(1 / 3600 * self.dt))

        self.trajectory.append((Balloon.x , Balloon.y, Balloon.altitude))
        self.time_history.append(self.timestamp)

        if self.total_steps > self.episode_length - 1:
            done = True
        else:
            done = False

        self.total_steps +=1

        return done

