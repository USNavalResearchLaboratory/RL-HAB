import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
import random
import time
from pynput import keyboard
from pynput.keyboard import Key, Controller
from stable_baselines3.common.env_util import make_vec_env
import math
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from line_profiler import LineProfiler
import sys

from env3d.generate3dflow import FlowField3D, PointMass
from env3d.rendering.renderer import MatplotlibRenderer

class FlowFieldEnv3d(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    # UPDATE: Now the enviornment takes in parameters we can keep track of.
    def __init__(self, x_dim = 500, y_dim = 500, z_dim = 100, min_vel =1, max_vel =10,
                 num_levels=6, dt=1, radius=100, max_accel=1.0, drag_coefficient=0.5, episode_length=400, decay_flow=False,
                 random_flow_episode_length=0,render_count=1, render_skip=100,seed=None, render_mode="human", alt_move = None):
        super(FlowFieldEnv3d, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.num_levels = num_levels # how many levels of different flow changes there are
        self.dt = dt
        self.radius = radius # station keeping radius
        self.radius_inner = radius*.5
        self.radius_outer = radius * 1.5

        self.max_accel = max_accel # acceleration in z-direction
        self.alt_move = alt_move

        self.drag_coefficient = drag_coefficient

        self.decay_flow = decay_flow #new feature

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

        self.FlowField3D = FlowField3D(self.x_dim, self.y_dim, self.z_dim, self.num_levels, self.min_vel, self.max_vel, self.res, seed)

        self.renderer = MatplotlibRenderer(x_dim=self.x_dim, y_dim=self.y_dim, z_dim=self.z_dim, FlowField3d = self.FlowField3D,
                                           render_count = self.render_count, render_skip = self.render_skip, render_mode = self.render_mode,
                                           radius = self.radius, dt = self.dt, episode_length = self.episode_length)

        self.state = {"mass":1,
                      "x":0, "y":0, "z":0,
                      "x_vel": 0, "y_vel": 0, "z_vel": 0}

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
            self.FlowField3D.randomize_flow()
            self.random_flow_episode_count = 0
            self.num_flow_changes +=1
        else:
            self.random_flow_episode_count +=1

        if self.decay_flow:
            self.FlowField3D.apply_boundary_decay(decay_type='linear')

        self.within_target = False
        self.twr = 0 # time within radius
        self.twr_inner = 0  # time within radius
        self.twr_outer = 0  # time within radius

        self.total_steps = 0

        #Make it discrete spawnings for now
        self.state["x"] = int(random.uniform(self.x_dim / 2 - self.radius_inner, self.x_dim / 2 + self.radius_inner))
        self.state["y"] = int(random.uniform(self.y_dim / 2 - self.radius_inner, self.y_dim / 2 + self.radius_inner))
        self.state["z"] = int(random.uniform(0 + self.z_dim / 4, self.z_dim - self.z_dim / 4))

        self.goal = {"x": self.x_dim/2,
                      "y": self.y_dim/2,
                     "z": 0}

        self.path = [(self.state["x"], self.state["y"], self.state["z"])]
        self.altitude_history = [self.state["z"]]

        self.renderer.reset(self.goal)

        return self._get_obs(), self._get_info()

    def move_agent(self, action):
        u, v, w = self.FlowField3D.interpolate_flow(int(self.state["x"]), int(self.state["y"]), int(self.state["z"]))

        #print(f"Current Flow Vel: {u}, {v}, {w}")
        ###print(f"Current Agent Vel: {self.state['x_vel']}, {self.state['y_vel']}, {self.state['z_vel']}")
        #print(f"Altitude: {self.state['z']}")


        if action == 2:  # up
            self.decelerate_flag = False
            input_accel_x, input_accel_y, input_accel_z = 0.0, 0.0, self.max_accel
        elif action == 0:  # down
            self.decelerate_flag = False
            input_accel_x, input_accel_y, input_accel_z = 0.0, 0.0, -self.max_accel
        else:  # stay
            #Need to perform a check to see if we need to deaccelerate or accelerate to 0 depending on our current motion
            if self.state["z_vel"] > 0:  # moving up, need to deaccelerate down to 0
                self.decelerate_flag = True
                self.decelerate_direction = -1  # decelerate down
                input_accel_x, input_accel_y, input_accel_z = 0.0, 0.0, -self.max_accel
            elif self.state["z_vel"] < 0:  # moving down, need to deaccelerate up to 0
                self.decelerate_flag = True
                self.decelerate_direction = 1  # decelerate up
                input_accel_x, input_accel_y, input_accel_z = 0.0, 0.0, self.max_accel
            else:  # already stationary, hold 0 velocity
                self.decelerate_flag = False
                self.decelerate_direction = 0
                input_accel_x, input_accel_y, input_accel_z = 0.0, 0.0, 0.0

        rel_vel_x = u - self.state["x_vel"]
        rel_vel_y = v - self.state['y_vel']
        rel_vel_z = w - self.state['z_vel']

        accel_x = (np.sign(rel_vel_x)*(self.drag_coefficient*(rel_vel_x**2))) + input_accel_x
        accel_y = (np.sign(rel_vel_y)*(self.drag_coefficient*(rel_vel_y**2))) + input_accel_y
        accel_z = (np.sign(rel_vel_z)*(self.drag_coefficient*(rel_vel_z**2))) + input_accel_z

        self.state["x"] = self.state["x"] + (self.state["x_vel"]*self.dt) + (0.5*accel_x*(self.dt**2))
        self.state["y"] = self.state["y"] + (self.state["y_vel"]*self.dt) + (0.5*accel_y*(self.dt**2))
        self.state["z"] = self.state["z"] + (self.state["z_vel"]*self.dt) + (0.5*accel_z*(self.dt**2))


        self.state["x_vel"] = self.state["x_vel"] + (accel_x*self.dt)
        self.state["y_vel"] = self.state["y_vel"] + (accel_y*self.dt)
        self.state["z_vel"] = np.clip(self.state["z_vel"] + (accel_z*self.dt),-self.alt_move,self.alt_move)

        # If deaccelerated past 0 velocity when stay has been called, hold at 0 velocity
        if self.decelerate_flag:
            if (self.decelerate_direction == -1 and self.state["z_vel"] < 0) or \
                    (self.decelerate_direction == 1 and self.state["z_vel"] > 0):
                self.state["z_vel"] = 0.0
                self.decelerate_flag = False  # Reset the flag once velocity is zero




        self.path.append((self.state["x"], self.state["y"], self.state["z"]))
        self.altitude_history.append(self.state["z"])

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

    def step(self, action):
        done = False
        self.total_steps += 1
        reward = self.move_agent(action)

        reward += self.reward_google()

        if self.total_steps > self.episode_length - 1:
            #reward += -100
            done = True
            #print("episode length", self.total_steps, "TWR", self._get_info()["twr"],
            #      "TWR_inner", self._get_info()["twr_inner"],
            #      "TWR_outer", self._get_info()["twr_outer"])

        observation = self._get_obs()
        info = self._get_info()

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
            rel_angle = self.calculate_relative_angle(self.state["x"], self.state["y"], self.goal["x"], self.goal["y"], flow_field_u[i], flow_field_v[i])
            magnitude = math.sqrt(flow_field_u[i]**2 + flow_field_v[i]**2)

            flow_field_rel_angle.append(rel_angle)
            flow_field_magnitude.append(magnitude)

        rel_flow_field = np.stack((flow_field_rel_angle, flow_field_magnitude), axis=-1)

        return rel_flow_field

    def _get_obs(self):

        distance = np.sqrt((self.state["x"] - self.goal["x"]) ** 2 + (self.state["y"] - self.goal["y"]) ** 2)
        rel_bearing = self.calculate_relative_angle(self.state["x"], self.state["y"], self.goal["x"], self.goal["y"], self.state["x_vel"], self.state["y_vel"])

        rel_flow_field = self.calculate_relative_flow_map()

        observation = {
            'altitude': np.array([self.state["z"]]),
            'distance': np.array([distance]),
            'rel_bearing': np.array([rel_bearing]),
            'flow_field': rel_flow_field
        }

        #print(distance, math.degrees(true_bearing), math.degrees(rel_bearing) )
        return observation

    def _get_info(self):
        return {
            "distance": np.sqrt((self.state["x"] - self.goal["x"])**2 + (self.state["y"] - self.goal["y"])**2),
            "within_target": self.within_target,
            "twr": self.twr,
            "twr_inner": self.twr_inner,
            "twr_outer": self.twr_outer,
            "num_flow_changes": self.num_flow_changes,
        }

    def render(self, mode='human'):
        self.renderer.render(mode='human', state = self.state, path = self.path, altitude_history = self.altitude_history)

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
    seed = None  # np.random.randint(0, 2 ** 32)  # Randomize random number generation, but kep the same across processes
    '''
    n_procs = 4

    # If you uncomment seeding,  the random path generation will be the same every time as long as seed is not None
    # np.random.seed(seed) #seeding

    env = make_vec_env(lambda: FlowFieldEnv3d(seed), n_envs=n_procs, seed=seed)

    # Initialize the PPO model with the environment
    model = DQN("MultiInputPolicy", env, seed=None, verbose=1, device='cpu')

    # Train the model
    model.learn(total_timesteps=10000)

    '''
    while True:
        dt = 60
        start_time = time.time()
        env_params = {
            'x_dim': 250,  # km
            'y_dim': 250,  # km
            'z_dim': 10,  # km
            'min_vel': 5 / 1000.,  # km/s
            'max_vel': 25 / 1000.,  # km/s
            'num_levels': 6,
            'dt': 60,  # seconds
            'radius': 50,  # km

            # DISCRETE
            'alt_move': 2 / 1000.,  # km/s  FOR DISCRETE

            # KINEMATICS
            'max_accel': 1.e-5,  # km/min^2
            'drag_coefficient': 0.5,

            'episode_length': 600,  # dt steps (minutes)
            'random_flow_episode_length': 1,  # how many episodes to regenerate random flow
            'decay_flow': False,
            'render_count': 1,
            'render_skip': 100,
            'render_mode': 'human',
            'seed': np.random.randint(0, 2 ** 32),
            # A random seed needs to be defined, to generated the same random numbers across processes
        }

        env = FlowFieldEnv3d(**env_params)
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

