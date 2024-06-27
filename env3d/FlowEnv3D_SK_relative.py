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

from generate3dflow import FlowField3D, PointMass

class FlowFieldEnv3d(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    # UPDATE: Now the enviornment takes in parameters we can keep track of.
    def __init__(self, x_dim = 500, y_dim = 500, z_dim = 100, min_vel =1, max_vel =10,
                 num_levels=6, dt=1, radius=100, alt_move=2, episode_length=400, decay_flow=False,
                 random_flow_episode_length=0,render_count=1, render_skip=100,seed=None, render_mode="human"):
        super(FlowFieldEnv3d, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.num_levels = num_levels # how many levels of different flow changes there are
        self.dt = dt
        self.radius = radius # station keeping radius
        self.alt_move = alt_move # how many units the agent can move up/down  (no kinematics)

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
        if self.random_flow_episode_count > self.random_flow_episode_length -1 and self.random_flow_episode_length !=0:
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

        if hasattr(self, 'fig'):
            plt.close(self.fig)
            delattr(self, 'fig')
            delattr(self, 'ax')
            delattr(self, 'ax2')
            delattr(self, 'ax3')
            delattr(self, 'goal')
            delattr(self, 'scatter')
            delattr(self, 'canvas')

        self.total_steps = 0

        #Make it discrete spawnings for now
        self.state["x"] = int(random.uniform(0 + self.x_dim/4, self.x_dim - self.x_dim/4))
        self.state["y"] = int(random.uniform(0 + self.y_dim/4, self.y_dim - self.y_dim/4))
        self.state["z"] = int(random.uniform(0 + self.z_dim/4, self.z_dim - self.z_dim/4))

        self.goal = {"x": self.x_dim/2,
                      "y": self.y_dim/2,
                     "z": 0}

        self.path = [(self.state["x"], self.state["y"], self.state["z"])]
        self.altitude_history = [self.state["z"]]

        self.render_step = 0

        return self._get_obs(), self._get_info()

    def move_agent(self, action):
        if action == 0:
            self.state["z"] = self.state["z"] - self.alt_move
        elif action == 2:
            self.state["z"] = self.state["z"] + self.alt_move

        # UDPATE: NO longer rounding to the nearest index value for interpolating
        u, v, _ = self.FlowField3D.interpolate_flow(self.state["x"], self.state["y"], self.state["z"])

        self.state["x"] = self.state["x"] + u * self.dt
        self.state["y"] = self.state["y"] + v * self.dt

        self.state["x_vel"] = u
        self.state["y_vel"] = v
        self.state["z_vel"] = self.alt_move*(action-1) #this can be done since theres only 3 actions

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

        return reward

    def step(self, action):
        done = False
        self.total_steps += 1
        reward = self.move_agent(action)

        reward += self.reward_google()

        if self.total_steps > self.episode_length - 1:
            #reward += -100
            done = True
            print("episode length", self.total_steps, "TWR", self._get_info()["twr"])

        if self.render_step == self.render_count:
            self.render_step = 0
        else:
            self.render_step += 1

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
            "num_flow_changes": self.num_flow_changes,
        }

    def plot_circle(self, ax, center_x,center_y, radius, plane='xy'):
        #UPDATE: This is a new function because the radius wasn't plotting properly for smaller radii
        # Create the angle array
        theta = np.linspace(0, 2 * np.pi, 100)

        # Generate the circle points in 2D
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)

        if plane == 'xy':
            x = center_x + circle_x
            y = center_y + circle_y
            z = np.full_like(x, 0)

        ax.plot(x, y, z, 'g--')

    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(18, 10))
            #self.ax = self.fig.add_subplot(231, projection='3d')
            #self.ax2 = self.fig.add_subplot(232, projection='3d')
            #self.ax3 = self.fig.add_subplot(212)

            gs = self.fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 4])
            self.ax3 = self.fig.add_subplot(gs[0, :])
            self.ax = self.fig.add_subplot(gs[1, 0], projection='3d')
            self.ax2 = self.fig.add_subplot(gs[1, 1], projection='3d')

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Altitude')
            self.ax.set_xlim(0, self.x_dim)
            self.ax.set_ylim(0, self.y_dim)
            self.ax.set_zlim(0, self.z_dim)

            self.path_plot, = self.ax.plot([], [], [], color='black')
            self.scatter = self.ax.scatter([], [], [], color='black')
            self.ground_track, = self.ax.plot([], [], [], color='red')
            self.scatter_goal = self.ax.scatter([], [], [], color='green')
            self.canvas = self.fig.canvas

            self.FlowField3D.visualize_3d_planar_flow(self.ax2, skip=self.render_skip)

            self.current_state_line, = self.ax.plot([], [], [], 'r--')
            #self.current_goal_line, = self.ax.plot([], [], [], 'g-')

            # Draw target circle on the XY plane
            self.plot_circle(self.ax,self.goal["x"],self.goal["y"], self.radius)

            self.altitude_line, = self.ax3.plot([], [], 'b-')
            self.ax3.set_xlabel('Number of Steps (dt=' + str(self.dt) + ')')
            self.ax3.set_ylabel('Altitude')
            self.ax3.set_xlim(0, self.episode_length)
            self.ax3.set_ylim(0, self.z_dim)

        if self.render_step == self.render_count:

            self.path_plot.set_data(np.array(self.path)[:, :2].T)
            self.path_plot.set_3d_properties(np.array(self.path)[:, 2])

            self.ground_track.set_data(np.array(self.path)[:, :2].T)
            self.ground_track.set_3d_properties(np.zeros(len(self.path)))

            self.scatter._offsets3d = (np.array([self.state["x"]]), np.array([self.state["y"]]), np.array([self.state["z"]]))
            self.scatter_goal._offsets3d = (np.array([self.goal["x"]]), np.array([self.goal["y"]]), np.array([0]))

            self.current_state_line.set_data([self.state["x"], self.state["x"]], [self.state["y"], self.state["y"]])
            self.current_state_line.set_3d_properties([0, self.state["z"]])

            self.altitude_line.set_data(range(len(self.altitude_history)), self.altitude_history)

            self.canvas.draw()
            #self.canvas.flush_events()

            if mode == 'human':
                plt.pause(0.001)

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

if __name__ == '__main__':
    seed = None #np.random.randint(0, 2 ** 32)  # Randomize random number generation, but kep the same across processes
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
        env_params = {
            'x_dim': 500,
            'y_dim': 500,
            'z_dim': 100,
            'min_vel': 5,
            'max_vel': 5,
            'num_levels': 6,
            'dt': 1,
            'radius': 100,
            'alt_move': 2,  # For discrete altitude moves
            'episode_length': 400,
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
        for step in range(400):
            # Use this for random action
            #action = env.action_space.sample()
            #obs, reward, done, _, info = env.step(action)

            #Use this for keyboard input
            obs, reward, done, truncated, info = env.step(last_action)
            print(obs)

            #print(step, reward)
            total_reward += reward
            if done:
                break
            env.render()
            #time.sleep(2)
        #print(obs)
        #print(env.FlowField3D.flow_field[:,0,0,0])
        print("Total reward:", total_reward, info, env_params['seed'])
