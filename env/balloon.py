
#from era5 import config_earth
import pandas as pd
import enum
from env.config.env_config import env_params
import numpy as np
import math

class BalloonState(object):
    """
    Represents the state of a high-altitude balloon during a simulation.

    :param x: Relative x position in meters (converted from latitude/longitude).
    :type x: float
    :param y: Relative y position in meters (converted from latitude/longitude).
    :type y: float
    :param altitude: Absolute altitude of the balloon in meters.
    :type altitude: float
    :param x_vel: Current velocity in the x direction (m/s).
    :type x_vel: float
    :param y_vel: Current velocity in the y direction (m/s).
    :type y_vel: float
    :param z_vel: Current ascent/descent velocity (m/s).
    :type z_vel: float
    :param lat: Current latitude of the balloon.
    :type lat: float
    :param lon: Current longitude of the balloon.
    :type lon: float
    :param distance: Distance to the station in the XY plane (meters).
    :type distance: float
    :param rel_bearing: Relative bearing of the motion direction with respect to the station.
    :type rel_bearing: float
    :param pressure: Atmospheric pressure around the balloon (currently unused).
    :type pressure: float
    """

    def __init__(self, x=None, y=None, altitude=None, x_vel=None, y_vel=None, z_vel=None,
                 distance=None, lat=None, lon=None, rel_bearing=None, pressure=None):
        #Trying to initialize to 0 to correct occasional error sith synth winds.
        self.x = x
        self.y = y
        self.altitude = altitude

        self.x_vel = x_vel
        self.y_vel = y_vel
        self.z_vel = z_vel

        self.lat = lat
        self.lon = lon

        self.distance= distance
        self.rel_bearing = rel_bearing

        self.atm_pressure = 0  #Figure this out later

        self.last_action= 0

        self.rel_wind_column= None #this might be the isssue

        #For later
        power= 0

    def __str__(self):
        """
        String representation of the balloon state.

        :returns: A formatted string showing all attributes of the balloon state.
        :rtype: str
        """
        return (f"BalloonState(x={self.x}, y={self.y}, altitude={self.altitude},\n "
                f"dist={self.distance}, rel_bearing={self.rel_bearing},\n "
                f"x_vel={self.x_vel}, y_vel={self.y_vel}, z_vel={self.z_vel},\n "
                f"lat={self.lat}, lon={self.lon}),\n"
                f"last_action={self.last_action}, rel_wind_column={self.rel_wind_column})")

    def update(self, **kwargs):
        """
        Update the attributes of the balloon state.

        :param kwargs: Key-value pairs representing attributes to update.
        :type kwargs: dict
        """
        for attr, value in kwargs.items():
            setattr(self, attr, value)


class AltitudeControlCommand(enum.IntEnum):
    """
    Enum representing altitude control commands.

    :cvar DOWN: Command to decrease altitude.
    :cvar STAY: Command to maintain current altitude.
    :cvar UP: Command to increase altitude.
    """

    DOWN = 0
    STAY = 1
    UP = 2

class SimulatorState(object):
    """
    Stores and updates overall state of the simulator.

    :param Balloon: The initial state of the balloon.
    :type Balloon: BalloonState
    :param timestamp: The start time of the simulation.
    :type timestamp: pd.Timestamp

    :ivar Balloon: Current state of the balloon.
    :vartype Balloon: BalloonState
    :ivar timestamp: Current simulation timestamp.
    :vartype timestamp: pd.Timestamp
    :ivar dt: Time step duration.
    :vartype dt: float
    :ivar total_steps: Total number of steps taken in the simulation.
    :vartype total_steps: int
    :ivar episode_length: Maximum number of steps in the simulation.
    :vartype episode_length: int
    :ivar trajectory: List of balloon positions (x, y, altitude) over time.
    :vartype trajectory: list
    :ivar time_history: List of timestamps corresponding to each step.
    :vartype time_history: list
    """
    def __init__(self,Balloon, timestamp):
        self.Balloon = Balloon

        self.timestamp = timestamp
        self.dt = env_params['dt']
        self.total_steps = 0

        self.episode_length = env_params['episode_length']

        self.trajectory = []
        self.time_history = []

    def step(self,Balloon):
        """
        Advance the simulator state by one step.

        :param Balloon: The current state of the balloon.
        :type Balloon: BalloonState
        :returns: True if the episode is complete, False otherwise.
        :rtype: bool
        """

        self.timestamp = self.timestamp + pd.Timedelta(hours=(1 / 3600 * self.dt))

        self.trajectory.append((Balloon.x , Balloon.y, Balloon.altitude))
        self.time_history.append(self.timestamp)

        if self.total_steps > self.episode_length - 1:
            done = True
        else:
            done = False

        self.total_steps += 1

        return done

