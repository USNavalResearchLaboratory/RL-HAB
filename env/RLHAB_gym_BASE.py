import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import time
from pynput import keyboard
import math
from env.rendering.renderertriple import MatplotlibRendererTriple
from env.config.env_config import env_params
from env.forecast_processing.forecast_visualizer import ForecastVisualizer
from env.balloon import BalloonState, SimulatorState
from env.balloon import AltitudeControlCommand as command
from env.forecast_processing.forecast import Forecast, Forecast_Subset
from env.forecast_processing.ForecastClassifier import ForecastClassifier
from env.rewards import reward_google, reward_piecewise, reward_euclidian, reward_bearing
from termcolor import colored

from utils.common import convert_range
from utils.initialize_forecast import initialize_forecasts, initialize_forecasts_full

class FlowFieldEnv3dBase(gym.Env):
    """
    A custom Gym environment simulating 3D flow fields for high-altitude balloons.

    This environment allows the agent to control the altitude of a balloon while considering
    two types of flow forecasts (ERA5 and synthetic). The goal is to perform station keeping
    within a defined radius around a target location.


    This base class needs to be extended by RLHAB_gym_SINGLE or RLHAB_gym_DUAL
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, days=1, render_mode=None):
        super().__init__()
        self.days = days
        self.render_mode = render_mode
        self.dt = env_params['dt']
        self.seed(env_params['seed'])

        self.ForecastClassifier = ForecastClassifier()

        self.radius = env_params['radius']
        self.radius_inner = self.radius * 0.5
        self.radius_outer = self.radius * 1.5

        self.goal = {"x": 0, "y": 0}
        self.renderer = None  # Set in child

        self.action_space = spaces.Discrete(3)

    def _build_obs_space(self, num_levels):
        min_vel, max_vel = 0, 50
        return spaces.Dict({
            'altitude': spaces.Box(low=env_params['alt_min'], high=env_params['alt_max'], shape=(1,), dtype=np.float64),
            'distance': spaces.Box(low=0, high=np.sqrt(env_params['rel_dist'] ** 2 + env_params['rel_dist'] ** 2), shape=(1,),
                                   dtype=np.float64),
            'rel_bearing': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
            'flow_field': spaces.Box(
                low=np.array([[env_params['alt_min'], 0, min_vel]] * num_levels, dtype=np.float64),
                high=np.array([[env_params['alt_max'], np.pi, max_vel]] * num_levels, dtype=np.float64),
                dtype=np.float64)
        })

    
    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.

        :param seed: Random seed value.
        :type seed: int, optional
        """
        if seed!=None:
            #print("Seed", seed)
            self.np_rng = np.random.default_rng(seed)
        else:
            self.np_rng = np.random.default_rng(np.random.randint(0, 2**32))

    def get_motion_forecast(self):
        raise NotImplementedError

    def get_flow_forecast(self):
        raise NotImplementedError
    
    #def initialize_balloon_and_sim(self, lat, lon, timestamp):
        
    
    def move_agent(self, action):
        """
        Update the balloon's position and altitude based on the selected action.

        Args:
            action (int): Action to take (0: Down, 1: Stay, 2: Up).

        Returns:
            float: reward (0 for now, no reward/penalty for moving).
        """

        movement_forecast = self.get_motion_forecast()
        flow_forecast = self.get_flow_forecast()

        # Look up flow at current 3D position before altitude change. Update Position and Flow State
        self.Balloon.lat,self.Balloon.lon,self.Balloon.x_vel,self.Balloon.y_vel, self.Balloon.x, self.Balloon.y = movement_forecast.getNewCoord(self.Balloon, self.SimulatorState, self.dt)

        self.Balloon.distance = np.sqrt((self.Balloon.x - self.goal["x"]) ** 2 + (self.Balloon.y - self.goal["y"]) ** 2)
        self.Balloon.rel_bearing = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                                 self.goal["x"], self.goal["y"],
                                                                 self.Balloon.x_vel, self.Balloon.y_vel)

        self.Balloon.rel_wind_column = self.calculate_relative_wind_column(flow_forecast)

        # Apply altitude change given action
        if action == command.UP:  # up
            # self.Balloon.z_vel = 2  # m/s
            self.Balloon.z_vel = np.random.normal(loc=env_params['ascent_rate_mean'],
                                                  scale=env_params['ascent_rate_std_dev'])
        elif action == command.DOWN:  # down
            # self.Balloon.z_vel = -3  # m/s
            self.Balloon.z_vel = -np.random.normal(loc=env_params['descent_rate_mean'],
                                                   scale=env_params['descent_rate_std_dev'])
        elif action == command.STAY:  # stay
            self.Balloon.z_vel = 0

        # Update Altitude and Last Action. Check for Altitude going out of bounds and clip.
        self.Balloon.altitude = np.clip(self.Balloon.altitude + self.Balloon.z_vel * self.dt,
                                        env_params["alt_min"],env_params["alt_max"])
        self.Balloon.last_action = action

        if self.Balloon.distance > env_params["rel_dist"]:
            if not self.rogue_status:
                self.rogue_step_trigger = self.total_steps

            self.rogue_count += 1
            self.rogue_status = True

        self.total_steps += 1

        return 0


    def baseline_controller(self, obs):
        """
        Given the current altitude and a list of relative flow column entries ([altitude, relative angle, speed]),
        this function returns the best altitude to transition to in order to minimize the relative angle and the action
        needed to reach that altitude.

        Args:
        - obs (dict): Observation dictionary that contains the following keys:
            - 'current_altitude' (float): The current altitude.
            - 'flow_field' (list of lists): List of flow field entries [altitude, relative angle, speed].

        Returns:
        - action (int): -1 for down, 0 for stay, 1 for up.
        - best_altitude (float): The altitude to transition to that minimizes the relative angle.
        """
        # Initialize variables to track the best altitude and its corresponding relative angle
        best_altitude = None
        min_relative_angle = float('inf')

        # Loop through the flow column to find the altitude with the smallest relative angle
        for level in obs['flow_field']:
            altitude, relative_angle, speed = level

            # Update if a new minimum relative angle is found
            # print(altitude, relative_angle, min_relative_angle)
            if relative_angle < min_relative_angle:
                min_relative_angle = relative_angle
                best_altitude = altitude

        # Determine the action to take based on the current altitude and the best altitude found
        if obs['altitude'] < best_altitude:
            action = 2  # Go up
        elif obs['altitude'] > best_altitude:
            action = 0  # Go down
        else:
            action = 1  # stay

        # Return the best altitude found
        return best_altitude, action

    
        
    def calculate_relative_angle(self, x, y, goal_x, goal_y, heading_x, heading_y):
        """
        Calculates the relative angle of motion of the blimp in relation to the goal based off of true bearing FROM POSITION TO GOAL
        and the true heading.  The relative angle of motion is then between 0 and 180 degrees, where...

        If the balloon moves directly to the goal at one altitude, the value would be 0 degrees.  Alternatively, if the balloon moved directly
        away from the goal at another altitude level, that would be 180 degrees.

        :param x: current balloon x position
        :param y: current balloon y position
        :param goal_x:
        :param goal_y:
        :param heading_x: current balloon x velocity
        :param heading_y: current balloon y velocity
        :return: relative bearing of balloon motion in relation to goal
        """

        # Calculate the current heading based on the heading vector
        heading = np.arctan2(heading_y, heading_x)

        # Calculate the true (inverted) bearing FROM POSITION TO GOAL
        true_bearing = np.arctan2(goal_y - y, goal_x - x)

        # Find the absolute difference between the two angles
        rel_bearing = np.abs(heading - true_bearing)

        #print(x, y, goal_x, goal_y, heading_x, heading_y)

        # map from [-pi, pi] to [0,pi]
        rel_bearing = abs((rel_bearing + np.pi) % (2 * np.pi) - np.pi)

        return rel_bearing

    def calculate_relative_wind_column(self, forecast_subset):
        """
        Builds off of the same calculation as calculate_relative_angle() to calculate the relative
        "flow map" vertical slice from the balloons current position between 0 and 180 degrees.

        (z, magnitude, bearing)  add uncertainty later)

        :return: flowfield with relative bearins and magnitude
        """

        #First need to get altitude coordinate from forecast
        #vertical_column = self.forecast.ds.sel(latitude=self.Balloon.lat, longitude=self.Balloon.lat, time= self.SimulatorState.timestamp, method='nearest')

        #alt_column = vertical_column['z'].values[::-1] / constants.GRAVITY
        #u_column = vertical_column['u'].values[::-1]
        #v_column = vertical_column['v'].values[::-1]

        alt_column,u_column,v_column = forecast_subset.np_lookup(self.Balloon.lat, self.Balloon.lon, self.SimulatorState.timestamp)

        alt_column = alt_column[::-1]
        u_column = u_column[::-1]
        v_column = v_column[::-1]

        flow_field_rel_angle = []
        flow_field_magnitude = []

        for i in range (0,len(alt_column)):
            rel_angle = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                      self.goal["x"], self.goal["y"], u_column[i], v_column[i])

            #print(alt_column[i])
            magnitude = math.sqrt(u_column[i]**2 + v_column[i]**2)

            flow_field_rel_angle.append(rel_angle)
            flow_field_magnitude.append(magnitude)

        #Relative Flow Field Map
        return np.stack((alt_column, flow_field_rel_angle, flow_field_magnitude), axis=-1)
    

    def _get_obs(self):
        """
        Get the current observation.

        Returns:
            dict: Observation dictionary.
        """
        observation = {
            'altitude': np.array([self.Balloon.altitude]),
            'distance': np.array([self.Balloon.distance]),
            'rel_bearing': np.array([self.Balloon.rel_bearing]),
            'flow_field': self.Balloon.rel_wind_column
        }

        return observation

    def _get_info(self):
        """
        Get additional environment info.

        Returns:
            dict: Information dictionary.
        """
        return {
            "distance": self.Balloon.distance,
            "within_target": self.within_target,
            "twr": self.twr,
            "twr_inner": self.twr_inner,
            "twr_outer": self.twr_outer,
            "forecast_score": self.forecast_score,
            "forecast_scores": self.forecast_scores,
            "render_mode": self.render_mode,
            "timestamp": self.timestamp,

            "total_steps": self.total_steps,
            "rogue_count": self.rogue_count,
            "rogue_step_trigger": self.rogue_step_trigger
        }
    
    def step(self, action):
        """
        Perform one step in the environment.

        :param action: Action to take (0: Down, 1: Stay, 2: Up).
        :type action: int
        :returns: Observation, reward, done flag, truncated flag, and additional info.
        :rtype: tuple
        """

        # For Baseline
        #best_altitude, baseline_action = self.baseline_controller(self._get_obs())
        #reward = self.move_agent(baseline_action)

        # For Normal Training with suggested actions
        reward = self.move_agent(action)
        observation = self._get_obs()
        info = self._get_info()

        # Initialize twr_data with current values
        twr_data = {
            "twr": self.twr,
            "twr_inner": self.twr_inner,
            "twr_outer": self.twr_outer
        }

        reward_step, self.within_target = reward_piecewise(
            self.Balloon, self.radius, self.radius_inner, self.radius_outer, twr_data)       
        reward += reward_step

        # Update instance tracking variables
        self.twr = twr_data["twr"]
        self.twr_inner = twr_data["twr_inner"]
        self.twr_outer = twr_data["twr_outer"]

        done = self.SimulatorState.step(self.Balloon)

        return observation, reward, done, False, info

    def render(self, mode='human'):
        """
        Render the environment.

        :param mode: Mode for rendering ('human', 'rgb_array', etc.).
        :type mode: str
        """
        
        if self.renderer:
            self.renderer.render(mode='human')

    def close(self):
        pass
