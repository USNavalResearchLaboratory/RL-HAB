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
from stable_baselines3 import DQN

from generate3dflow import FlowField3D, PointMass

class FlowFieldEnv3d(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, seed=None, render_mode="human"):
        super(FlowFieldEnv3d, self).__init__()
        self.x_dim = 500
        self.y_dim = 500
        self.z_dim = 100
        self.min_vel = 1
        self.max_vel = 10
        self.num_levels = 6  # how many levels of different flow changes there are
        self.dt = 1
        self.radius = 100  # station keeping radius
        self.alt_move = 2  # how many units the agent can move up/down  (no kinematics)

        # Counting Defaults
        self.num_flow_changes = 0  # do not change from 0
        self.random_flow_episode_count = 0  # do not change from 0
        self.total_steps = 0  # do not change from 0

        self.episode_length = 400  # how long an episode is
        self.random_flow_episode_length = 0  # how many episodes before randomizing flow field
        self.render_count = 1  # how many steps before rendering

        self.render_mode = render_mode

        self.seed(seed)

        self.FlowField3D = FlowField3D(self.x_dim, self.y_dim, self.z_dim, self.num_levels, self.min_vel, self.max_vel,
                                       seed)

        self.state = {"mass":1,
                      "x":0, "y":0, "z":0,
                      "x_vel": 0, "y_vel": 0, "z_vel": 0}

        self.total_steps = 0

        self.action_space = spaces.Discrete(3)  # 0: Move down, 1: Stay, 2: Move up

        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=0, high=self.x_dim, shape=(1,), dtype=np.float64),
            'y': spaces.Box(low=0, high=self.y_dim, shape=(1,), dtype=np.float64),
            'z': spaces.Box(low=0, high=self.z_dim, shape=(1,), dtype=np.float64),
            'x_vel': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'y_vel': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'z_vel': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'goal_x': spaces.Box(low=0, high=self.x_dim, shape=(1,), dtype=np.float64),
            'goal_y': spaces.Box(low=0, high=self.y_dim, shape=(1,), dtype=np.float64),
            'goal_z': spaces.Box(low=0, high=self.z_dim, shape=(1,), dtype=np.float64),
            'flow_field': spaces.Box(low=-self.max_vel, high=self.max_vel, shape=(self.num_levels, 2), dtype=np.float64)
        })

    def seed(self, seed=None):
        if seed != None:
            self.np_rng = np.random.default_rng(seed)
        else:
            self.np_rng = np.random.default_rng(np.random.randint(0, 2 ** 32))

    def reset(self, seed=None, options=None):
        if self.random_flow_episode_count > self.random_flow_episode_length -1 and self.random_flow_episode_length !=0:
            #self.FlowField3D.generate_random_planar_flow_field()
            #self.FlowField3D.gradualize_random_flow()
            self.FlowField3D.randomize_flow()
            self.random_flow_episode_count = 0
            self.num_flow_changes +=1
        else:
            self.random_flow_episode_count +=1


        #self.FlowField3D.generate_random_planar_flow_field()
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
                      "y": self.x_dim/2,
                     "z": 0}

        self.path = [(self.state["x"], self.state["y"], self.state["z"])]
        self.altitude_history = [self.state["z"]]

        self.render_count = 10
        self.render_step = 0

        return self._get_obs(), self._get_info()

    def move_agent(self, action):
        if action == 0:
            self.state["z"] = self.state["z"] - self.alt_move
        elif action == 2:
            self.state["z"] = self.state["z"] + self.alt_move

        u, v, _ = self.FlowField3D.interpolate_flow(int(self.state["x"]), int(self.state["y"]), int(self.state["z"]))

        self.state["x"] = self.state["x"] + u * self.dt
        self.state["y"] = self.state["y"] + v * self.dt

        self.state["x_vel"] = u
        self.state["y_vel"] = v
        self.state["z_vel"] = self.alt_move*(action-1) #this can be done since theres only 3 actions

        self.path.append((self.state["x"], self.state["y"], self.state["z"]))
        self.altitude_history.append(self.state["z"])

        return 0


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

        #Check if agent has gone out of bounds?
        '''
        if self.state["x"] < 0 or self.state["x"] > self.x_dim or \
                self.state["y"] < 0 or self.state["y"] > self.y_dim or \
                self.state["z"] < 0 or self.state["z"] > self.z_dim:
            reward += -100
            done = True
        '''

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

    def _get_obs(self):
        # Extract u and v components of the flow field at each altitude level
        #Maybe change this to only happen once?
        flow_field_u = self.FlowField3D.flow_field[:, 0, 0, 0]
        flow_field_v = self.FlowField3D.flow_field[:, 0, 0, 1]

        flow_field = np.stack((flow_field_u, flow_field_v), axis=-1)



        observation = {
            'x': np.array([self.state["x"]]),
            'y': np.array([self.state["y"]]),
            'z': np.array([self.state["z"]]),
            'x_vel': np.array([self.state["x_vel"]]),
            'y_vel': np.array([self.state["y_vel"]]),
            'z_vel': np.array([self.state["z_vel"]]),
            'goal_x': np.array([self.goal["x"]]),
            'goal_y': np.array([self.goal["y"]]),
            'goal_z': np.array([0]),
            'flow_field': flow_field
        }
        return observation

    def _get_info(self):
        return {
            "distance": np.sqrt((self.state["x"] - self.goal["x"])**2 + (self.state["y"] - self.goal["y"])**2),
            "within_target": self.within_target,
            "twr": self.twr,
            "num_flow_changes": self.num_flow_changes,

        }

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

            self.FlowField3D.visualize_3d_planar_flow(self.ax2, skip=50)

            self.current_state_line, = self.ax.plot([], [], [], 'r--')
            #self.current_goal_line, = self.ax.plot([], [], [], 'g-')

            # Draw target circle on the XY plane
            theta = np.linspace(0, 2*np.pi, self.radius)
            x_circle = self.goal["x"] + self.radius * np.cos(theta)
            y_circle = self.goal["y"] + self.radius * np.sin(theta)
            z_circle = np.zeros_like(x_circle)
            self.ax.plot(x_circle, y_circle, z_circle, 'g--')

            self.altitude_line, = self.ax3.plot([], [], 'b-')
            self.ax3.set_xlabel('Time Step')
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

            #self.current_goal_line.set_data([self.goal["x"], self.goal["x"]], [self.goal["y"], self.goal["y"]])
            #self.current_goal_line.set_3d_properties([0, self.z_dim])

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
last_action = 0  # Default action
listener = keyboard.Listener(on_press=on_press)
listener.start()

if __name__ == '__main__':
    seed = np.random.randint(0, 2 ** 32) #Randomize random number generation, but kep the same across processes
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
        env.reset()
        total_reward = 0
        for step in range(400):
            action = env.action_space.sample()
            # Use this for random action
            obs, reward, done, _, info = env.step(action)

            #Use this for keyboard input
            #obs, reward, done, truncated, info = env.step(last_action)

            #print(step, reward)
            total_reward += reward
            if done:
                break
            env.render()
        #print(obs)
        #print(env.FlowField3D.flow_field[:,0,0,0])
        print("Total reward:", total_reward, info)
    '''