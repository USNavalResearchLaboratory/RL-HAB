import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import time
from pynput import keyboard
import math
import pandas as pd

from env3d.generate3dflow import FlowField3D, PointMass
from env3d.rendering.renderer import MatplotlibRenderer
from utils.convert_range import convert_range
from utils import constants
from env3d.config.env_config import env_params
from era5.forecast_visualizer import ForecastVisualizer
from env3d.balloon import BalloonState, SimulatorState
from env3d.balloon import AltitudeControlCommand as command
from era5.forecast import Forecast

from era5 import config_earth
from era5 import ERA5

class FlowFieldEnv3d(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    # UPDATE: Now the enviornment takes in parameters we can keep track of.
    def __init__(self, seed=None, render_mode=None):
        super(FlowFieldEnv3d, self).__init__()

        self.radius = env_params['radius'] # station keeping radius
        self.radius_inner = self.radius *.5
        self.radius_outer = self.radius * 1.5

        self.rel_dist = env_params['rel_dist']


        # Import configuration file variables
        self.coord = config_earth.simulation['start_coord']
        self.start_time = config_earth.simulation['start_time']
        self.dt = config_earth.simulation['dt']

        self.gfs = ERA5.ERA5(self.coord)

        self.render_mode = render_mode

        self.seed(seed)
        self.res = 1

        self.Forecast_visualizer = ForecastVisualizer()
        self.Forecast_visualizer.generate_flow_array(self.start_time) #change this for time?


        if self.render_mode=="human":
            self.renderer = MatplotlibRenderer(
                                               Forecast_visualizer=self.Forecast_visualizer,
                                               render_mode=self.render_mode, radius=self.radius,  coordinate_system = "geographic")

        self.action_space = spaces.Discrete(3)  # 0: Move down, 1: Stay, 2: Move up

        # Need to reset the forecast here?
        # WRITE A FORECAST RANDOMIZER LEVEL
        self.forecast = Forecast(self.rel_dist, env_params['pres_min'], env_params['pres_max'])

        min_vel = 0
        max_vel = 50
        num_levels = len(self.forecast.pressure_levels)

        # These number ranges are technically wrong.  Velocity could be 0?  Altitude can currently go out of bounds.
        self.observation_space = spaces.Dict({
            'altitude': spaces.Box(low=env_params['alt_min'], high=env_params['alt_max'], shape=(1,), dtype=np.float64),
            'distance': spaces.Box(low=0, high=np.sqrt(self.rel_dist ** 2 + self.rel_dist ** 2), shape=(1,),
                                   dtype=np.float64),
            'rel_bearing': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),

            # This will be based off of Pressure levels
            # (Altitude, bearing, Magnitude)  * mandatory pressure levels of forecast
            'flow_field': spaces.Box(
                low=np.array([[env_params['alt_min'], 0, min_vel]] * num_levels, dtype=np.float64),
                high=np.array([[env_params['alt_max'], np.pi, max_vel]] * num_levels, dtype=np.float64),
                dtype=np.float64)
        })

    def seed(self, seed=None):
        if seed!=None:
            self.np_rng = np.random.default_rng(seed)
        else:
            self.np_rng = np.random.default_rng(np.random.randint(0, 2**32))

    def reset(self, seed=None, options=None):


        self.within_target = False
        self.twr = 0 # time within radius
        self.twr_inner = 0  # time within radius
        self.twr_outer = 0  # time within radius


        self.goal = {"x": 0,
                      "y": 0,
                     "z": 0}

        self.Balloon = BalloonState(lat = self.coord['lat'],
                                    lon = self.coord['lon'],

                                    x = 0,
                                    y = 0,
                                    altitude = int(random.uniform(env_params['alt_min'],env_params['alt_max']))
                                    )

        self.SimulatorState = SimulatorState(self.Balloon)

        #Do an artificial move to get some initial vleocity, disntance, and bearing values, then reset back to initial coordinates
        self.move_agent(1)
        self.Balloon.update(lat = self.coord['lat'],lon = self.coord['lon'],x=0,y=0, distance = 0)

        if self.render_mode == "human":
            self.renderer.reset(self.goal, self.Balloon, self.SimulatorState )

        return self._get_obs(), self._get_info()

    def move_agent(self, action):

        #Update Agent Movement  (NEED TO CHANGE TO HERE

        #Need to look up flow at current position then integrate forward


        #simulate a coord for ERA5:
        coord = {
            "lat": self.Balloon.lat,  # (deg) Latitude
            "lon": self.Balloon.lon,  # (deg) Longitude
            "alt": self.Balloon.altitude,  # (m) Elevation
            "timestamp": self.SimulatorState.timestamp,  # Timestamp
        }

        self.Balloon.lat,self.Balloon.lon,self.Balloon.x_vel,self.Balloon.y_vel, _, _, _,_, _, _ = self.gfs.getNewCoord(coord,self.dt)


        #Take care of Actions
        if action == command.UP:  # up
            self.Balloon.z_vel = 2  # m/s
        elif action == command.DOWN:  # down
            self.Balloon.z_vel = -3  # m/s
        elif action == command.STAY:  # stay
            self.Balloon.z_vel = 0



        self.Balloon.update(x=self.Balloon.x + self.Balloon.x_vel * self.dt,
                            y=self.Balloon.y + self.Balloon.y_vel * self.dt,
                            altitude=self.Balloon.altitude + self.Balloon.z_vel * self.dt,

                            last_action = action
                            )

        self.Balloon.distance = np.sqrt((self.Balloon.x - self.goal["x"]) ** 2 + (self.Balloon.y - self.goal["y"]) ** 2)

        self.Balloon.rel_bearing = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                                 self.goal["x"], self.goal["y"],
                                                                 self.Balloon.x_vel, self.Balloon.y_vel)


        return 0 #No reward or penalty for moving for now

    def reward_google(self):
        distance_to_target = np.sqrt((self.state["x"] - self.goal["x"]) ** 2 + (self.state["y"] - self.goal["y"]) ** 2)
        c_cliff = 0.4
        tau = 100

        if distance_to_target <= self.radius:
            reward = 1
            self.twr +=1
            self.within_target = True
        else:
            #reward = np.exp(-0.01 * (distance_to_target - self.radius))
            reward = c_cliff*2*np.exp((-1*(distance_to_target-self.radius)/tau))
            self.within_target = False


        #Add more regions to track,  Not doing anything with them yet,  just for metric analysis
        if distance_to_target <= self.radius_inner:
            self.twr_inner += 1

        if distance_to_target <= self.radius_outer:
            self.twr_outer += 1

        return reward

    def reward_piecewise(self):
        '''
        Extra reward possible for station keeping within inner necessary, otherwise following google's structure
        :return:
        '''
        distance_to_target = np.sqrt((self.state["x"] - self.goal["x"]) ** 2 + (self.state["y"] - self.goal["y"]) ** 2)
        c_cliff = 0.4
        tau = 100

        if distance_to_target <= self.radius_inner:
            reward = 2
            self.twr += 1
            self.within_target = True
        elif distance_to_target <= self.radius and distance_to_target > self.radius_inner:
            reward = 1
            self.twr += 1
            self.within_target = True
        else:
            # reward = np.exp(-0.01 * (distance_to_target - self.radius))
            reward = c_cliff * 2 * np.exp((-1 * (distance_to_target - self.radius) / tau))
            self.within_target = False

        # Add more regions to track,  Not doing anything with them yet,  just for metric analysis
        if distance_to_target <= self.radius_inner:
            self.twr_inner += 1

        if distance_to_target <= self.radius_outer:
            self.twr_outer += 1

        return reward

    def reward_euclidian(self):
        '''
        Linear Euclidian reward within target region, google cliff function for outside of radius

        :return:
        '''


        distance_to_target = self.Balloon.distance
        c_cliff = 0.4
        tau = 100

        if distance_to_target <= self.radius:
            #Normalize distance within radius,  for a maximum score of 2.
            reward = convert_range(distance_to_target,0,self.radius, 2, 1)
            self.twr += 1
            self.within_target = True

        else:
            # reward = np.exp(-0.01 * (distance_to_target - self.radius))
            reward = c_cliff * 2 * np.exp((-1 * (distance_to_target - self.radius) / tau))
            self.within_target = False

        # Add more regions to track,  Not doing anything with them yet,  just for metric analysis
        if distance_to_target <= self.radius_inner:
            self.twr_inner += 1

        if distance_to_target <= self.radius_outer:
            self.twr_outer += 1

        return reward

    def step(self, action):
        reward = self.move_agent(action)
        reward += self.reward_euclidian()

        self.calculate_relative_wind_column()

        observation = self._get_obs()
        info = self._get_info()

        done = self.SimulatorState.step(self.Balloon)


        return observation, reward, done, False, info

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

        # map from [-pi, pi] to [0,pi]
        rel_bearing = abs((rel_bearing + np.pi) % (2 * np.pi) - np.pi)

        return rel_bearing

    def calculate_relative_wind_column(self):
        """
        Builds off of the same calculation as calculate_relative_angle() to calculate the relative
        "flow map" vertical slice from the balloons current position between 0 and 180 degrees.

        (z, magnitude, bearing)  add uncertatinty later)

        :return: flowfield with relative bearins and magnitude
        """

        #First need to get altitude coordinate from forecast
        vertical_column = self.forecast.ds.sel(latitude=self.Balloon.lat, longitude=self.Balloon.lat, time= self.SimulatorState.timestamp, method='nearest')

        alt_column = vertical_column['z'].values[::-1] / constants.GRAVITY
        u_column = vertical_column['u'].values[::-1]
        v_column = vertical_column['v'].values[::-1]

        flow_field_rel_angle = []
        flow_field_magnitude = []

        for i in range (0,len(alt_column)):
            rel_angle = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                      self.goal["x"], self.goal["y"], u_column[i], v_column[i])
            magnitude = math.sqrt(u_column[i]**2 + v_column[i]**2)

            flow_field_rel_angle.append(rel_angle)
            flow_field_magnitude.append(magnitude)

        #Relative Flow Field Map
        self.Balloon.rel_wind_column = np.stack((alt_column, flow_field_rel_angle, flow_field_magnitude), axis=-1)

    def _get_obs(self):
        observation = {
            'altitude': np.array([self.Balloon.altitude]),
            'distance': np.array([self.Balloon.distance]),
            'rel_bearing': np.array([self.Balloon.rel_bearing]),
            'flow_field': self.Balloon.rel_wind_column
        }

        return observation

    def _get_info(self):
        return {
            "distance": self.Balloon.distance,
            "within_target": self.within_target,
            "twr": self.twr,
            "twr_inner": self.twr_inner,
            "twr_outer": self.twr_outer,
        }

    def render(self, mode='human'):
        self.renderer.render(mode='human')

    def close(self):
        pass

#for keyboard control
def on_press(key):
    global last_action
    try:
        if key == keyboard.KeyCode.from_char("w"):
            last_action = command.UP
        elif key == keyboard.KeyCode.from_char("d"):
            last_action = command.STAY
        elif key == keyboard.KeyCode.from_char("x"):
            last_action = command.DOWN
        else:
            last_action = command.STAY  # Default action when no arrow key is pressed
    except AttributeError:
        pass

# Global variable to store the last action pressed
last_action = command.STAY  # Default action
listener = keyboard.Listener(on_press=on_press)
listener.start()

def main():
    while True:
        start_time = time.time()

        env = FlowFieldEnv3d(render_mode="human")
        env.reset()
        total_reward = 0
        for step in range( env_params["episode_length"]):
            # Use this for random action
            # action = env.action_space.sample()
            # obs, reward, done, _, info = env.step(action)

            # Use this for keyboard input
            obs, reward, done, truncated, info = env.step(last_action)
            # print(obs)

            # print(step, reward)
            total_reward += reward
            if done:
                break
            env.render()
            # time.sleep(2)
        # print(obs)
        # print(env.FlowField3D.flow_field[:,0,0,0])
        print("Total reward:", total_reward, info, env_params['seed'])

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time)


        #sys.exit()


if __name__ == '__main__':
    main()
