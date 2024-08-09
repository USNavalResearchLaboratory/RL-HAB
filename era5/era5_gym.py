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
from env3d.config.env_config import env_params
from era5.forecast_visualizer import ForecastVisualizer
from env3d.balloon import BalloonState, SimulatorState
from env3d.balloon import AltitudeControlCommand as command

import config_earth
import ERA5

class FlowFieldEnv3d(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    # UPDATE: Now the enviornment takes in parameters we can keep track of.
    def __init__(self, x_dim = 500, y_dim = 500, z_dim = 100, min_vel =1, max_vel =10,
                 num_levels=6, dt=1, radius=100, max_accel=1.0, drag_coefficient=0.5, episode_length=400, decay_flow=False,
                 random_flow_episode_length=0,render_count=1, render_skip=100,seed=None, render_mode="human", alt_move = None,
                 alt_min = 15000, alt_max = 28000):
        super(FlowFieldEnv3d, self).__init__()


        self.radius = radius # station keeping radius
        self.radius_inner = radius*.5
        self.radius_outer = radius * 1.5


        # Import configuration file variables
        self.coord = config_earth.simulation['start_coord']
        start = config_earth.simulation['start_time']
        self.t = start
        min_alt = env_params['alt_min']  #this was switched to env paramaters
        float = config_earth.simulation['float']
        self.dt = config_earth.simulation['dt']
        sim = config_earth.simulation["sim_time"]
        GFSrate = config_earth.forecast["GFSrate"]
        self.gfs = ERA5.ERA5(self.coord)

        # Initialize trajectory variables
        #self.el = [min_alt]  # 9000
        #self.el_new = min_alt
        self.coords = [self.coord]
        self.lat = [self.coord["lat"]]
        self.lon = [self.coord["lon"]]
        self.ttt = [self.t]


        # Counting Defaults
        self.num_flow_changes = 0 # do not change from 0
        self.random_flow_episode_count = 0  # do not change from 0
        self.total_steps = 0 # do not change from 0

        self.episode_length = episode_length #how long an episode is
        self.random_flow_episode_length = random_flow_episode_length # how many episodes before randomizing flow field

        self.render_count = render_count #how many steps before rendering
        self.render_skip = render_skip
        self.render_mode = render_mode

        self.seed(seed)
        self.res = 1

        self.FlowField3D = ForecastVisualizer()
        self.FlowField3D.generate_flow_array(0)

        ####### OLD STUFF
        self.x_dim = -40
        self.y_dim = 100
        self.z_dim = 1000
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.num_levels = num_levels
        ####################

        self.renderer = MatplotlibRenderer(
                                           FlowField3d=self.FlowField3D,
                                           render_count=self.render_count, render_skip=self.render_skip,
                                           render_mode=self.render_mode, radius=self.radius,
                                           episode_length=self.episode_length, coordinate_system = "geographic")

        self.action_space = spaces.Discrete(3)  # 0: Move down, 1: Stay, 2: Move up

        #These number ranges are technically wrong.  Velocity could be 0?  Altitude can currently go out of bounds.
        self.observation_space = spaces.Dict({
            'altitude': spaces.Box(low=0, high=self.z_dim, shape=(1,), dtype=np.float64),
            'distance': spaces.Box(low=0, high=np.sqrt(self.x_dim ** 2 + self.y_dim ** 2), shape=(1,),
                                       dtype=np.float64),
            'rel_bearing': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
            'flow_field': spaces.Box(
                                low=np.array([[0, self.min_vel]] * self.num_levels, dtype=np.float64),
                                high=np.array([[np.pi, self.max_vel]] * self.num_levels, dtype=np.float64),
                                dtype=np.float64)
        })

    def seed(self, seed=None):
        if seed!=None:
            self.np_rng = np.random.default_rng(seed)
        else:
            self.np_rng = np.random.default_rng(np.random.randint(0, 2**32))

    def reset(self, seed=None, options=None):
        if self.random_flow_episode_count >= self.random_flow_episode_length -1 and self.random_flow_episode_length !=0:
            #self.FlowField3D.generate_random_planar_flow_field()
            #self.FlowField3D.gradualize_random_flow()
            #self.FlowField3D.randomize_flow()
            self.random_flow_episode_count = 0
            self.num_flow_changes +=1
        else:
            self.random_flow_episode_count +=1

        #if self.decay_flow:
        #    self.FlowField3D.apply_boundary_decay(decay_type='linear')



        self.within_target = False
        self.twr = 0 # time within radius
        self.twr_inner = 0  # time within radius
        self.twr_outer = 0  # time within radius

        self.total_steps = 0


        self.goal = {"x": 0,
                      "y": 0,
                     "z": 0}


        self.Balloon = BalloonState(lat = self.coord['lat'],
                                    lon = self.coord['lon'],

                                    x = 0,
                                    y = 0,
                                    z = int(random.uniform(env_params['alt_min'],env_params['alt_max']))
                                    )

        self.SimulatorState = SimulatorState(self.Balloon)

        #Do an artificial move to get some initial vleocity, disntance, and bearing values, then reset back to initial coordinates
        self.move_agent(1)
        self.Balloon.update(lat = self.coord['lat'],lon = self.coord['lon'],x=0,y=0, distance = 0)

        self.renderer.reset(self.goal, self.Balloon, self.SimulatorState )

        return self._get_obs(), self._get_info()

    def move_agent(self, action):

        #Update Agent Movement  (NEED TO CHANGE TO HERE

        #Need to look up flow at current position then integrate forward


        #simulate a coord for ERA5:
        coord = {
            "lat": self.Balloon.lat,  # (deg) Latitude
            "lon": self.Balloon.lon,  # (deg) Longitude
            "alt": self.Balloon.z,  # (m) Elevation
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


        '''
        self.el.append(self.Balloon.z)
        self.coords.append(coord_new)
        self.lat.append(self.Balloon.lat)
        self.lon.append(self.Balloon.lon)
        self.ttt.append(self.t)
        '''


        self.Balloon.update(x=self.Balloon.x + self.Balloon.x_vel * self.dt,
                            y=self.Balloon.y + self.Balloon.y_vel * self.dt,
                            z=self.Balloon.z + self.Balloon.z_vel * self.dt,

                            last_action = action
                            )

        self.Balloon.distance = np.sqrt((self.Balloon.x - self.goal["x"]) ** 2 + (self.Balloon.y - self.goal["y"]) ** 2)

        self.Balloon.rel_bearing = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                                 self.goal["x"], self.goal["y"],
                                                                 self.Balloon.x_vel, self.Balloon.y_vel)



        #self.path.append((self.Balloon.x, self.Balloon.y, self.Balloon.z))
        #self.altitude_history.append(self.Balloon.z)

        #update timestamp
        self.t = self.t + pd.Timedelta(hours=(1 / 3600 * self.dt))

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
        done = False

        reward = self.move_agent(action)
        reward += self.reward_euclidian()

        if self.total_steps > self.episode_length - 1:
            done = True

        observation = self._get_obs()
        info = self._get_info()

        self.total_steps += 1

        self.SimulatorState.step(self.Balloon)

        #print(info["distance"], reward)

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

    def calculate_relative_flow_map(self):
        """
        Builds off of the same calculation as calculate_relative_angle() to caluclate the relative Calculates the relative
        "flow map" vertical slice from the balloons current position between 0 and 180 degrees.

        :return: flowfield with relative bearins and magnitude
        """

        flow_field_u = self.FlowField3D.flow_field[:, 0, 0, 0]
        flow_field_v = self.FlowField3D.flow_field[:, 0, 0, 1]

        flow_field_rel_angle = []
        flow_field_magnitude = []

        for i in range (0,len(flow_field_u)):
            rel_angle = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y, self.Balloon.z, self.goal["y"], flow_field_u[i], flow_field_v[i])
            magnitude = math.sqrt(flow_field_u[i]**2 + flow_field_v[i]**2)

            flow_field_rel_angle.append(rel_angle)
            flow_field_magnitude.append(magnitude)

        rel_flow_field = np.stack((flow_field_rel_angle, flow_field_magnitude), axis=-1)

        return rel_flow_field

    def _get_obs(self):

        #distance = np.sqrt((self.state["x"] - self.goal["x"]) ** 2 + (self.state["y"] - self.goal["y"]) ** 2)
        #rel_bearing = self.calculate_relative_angle(self.state["x"], self.state["y"], self.goal["x"], self.goal["y"], self.state["x_vel"], self.state["y_vel"])

        rel_flow_field = self.calculate_relative_flow_map()

        observation = {
            'altitude': np.array([self.Balloon.z]),
            'distance': np.array([self.Balloon.distance]),
            'rel_bearing': np.array([self.Balloon.rel_bearing]),
            'flow_field': rel_flow_field
        }

        #print(distance, math.degrees(true_bearing), math.degrees(rel_bearing) )
        return observation

    def _get_info(self):
        return {
            "distance": self.Balloon.distance,
            "within_target": self.within_target,
            "twr": self.twr,
            "twr_inner": self.twr_inner,
            "twr_outer": self.twr_outer,
            "num_flow_changes": self.num_flow_changes,
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
            last_action = 2
        elif key == keyboard.KeyCode.from_char("d"):
            last_action = 1
        elif key == keyboard.KeyCode.from_char("x"):
            last_action = 0
        else:
            last_action = 0  # Default action when no arrow key is pressed
    except AttributeError:
        pass

# Global variable to store the last action pressed
last_action = 1  # Default action
listener = keyboard.Listener(on_press=on_press)
listener.start()



def main():
    while True:
        start_time = time.time()

        env = FlowFieldEnv3d(**env_params)
        env.reset()
        total_reward = 0
        for step in range( env_params["episode_length"]):
            # Use this for random action
            # action = env.action_space.sample()
            # obs, reward, done, _, info = env.step(action)

            #time.sleep(1)
            print(env.SimulatorState.timestamp)
            print(env.Balloon)
            print()

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

