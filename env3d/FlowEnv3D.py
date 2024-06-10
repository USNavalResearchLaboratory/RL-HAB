import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
import random
import time

#from stable_baselines3.common.env_checker import check_env

from generate3dflow import FlowField3D

class FlowFieldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode="human"):
        self.x_dim = 500
        self.y_dim = 500
        self.z_dim = 100
        self.min_vel = 1
        self.max_vel = 10
        self.num_levels = 6
        self.dt = 1

        #self.max_vel = 10
        #self.min_vel = 1
        self.episode_length = 400
        self.alt_move = 1

        self.render_mode = render_mode

        self.FlowField3D = FlowField3D(self.x_dim, self.y_dim, self.z_dim, self.num_levels, self.min_vel, self.max_vel, seed =random.randint(0, 10000))


        self.state = {"mass":1,
                      "x":0, "y":0, "z":0,
                      "x_vel": 0, "y_vel": 0, "z_vel": 0}

        self.total_steps = 0

        self.action_space = spaces.Discrete(3)  # 0: Move down, 1: Stay, 2: Move up

        # Observation space includes continuous x and y positions and discrete flow field
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=0, high=self.x_dim, shape=(1,), dtype=np.float64),
            'y': spaces.Box(low=0, high=self.y_dim, shape=(1,), dtype=np.float64),
            'z': spaces.Box(low=0, high=self.z_dim, shape=(1,), dtype=np.float64),
            'x_vel': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'y_vel': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'z_vel': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),\
            #Need to add in Flow Map
            'goal_x': spaces.Box(low=0, high=self.x_dim, shape=(1,), dtype=np.float64),
            'goal_y': spaces.Box(low=0, high=self.y_dim, shape=(1,), dtype=np.float64),
            'goal_z': spaces.Box(low=0, high=self.z_dim, shape=(1,), dtype=np.float64),
        })


    def reset(self, seed=None, options=None):

        #Different options for randomizing flow (uncomment which option you want. Default is randomize every episode)
        #                                               # 1. Do nothing
        #self.FlowField3D.gradualize_random_flow()      # 2. Gradually Randomize the flow every episode (up to 10- degrees per level)
        self.FlowField3D.randomize_flow()               # 3. Randomize the flor every episode from directional bins [0, pi/2, pi, 3pi/2]

        #self.FlowField3D.generate_random_planar_flow_field()
        self.target_reached = False

        # Reset the rendering
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            delattr(self, 'fig')
            delattr(self, 'ax')
            delattr(self, 'ax2')
            #delattr(self, 'im')
            delattr(self, 'goal')
            delattr(self, 'scatter')
            delattr(self, 'canvas')

        self.total_steps = 0

        # Reset point position and include goal position in the observation
        self.state["x"] = 250 #random.uniform(0 +20, self.x_dim-20)
        self.state["y"] = 250 #random.uniform(0 +20, self.y_dim-20)
        self.state["z"] = 0 #random.uniform(0 +20, self.z_dim-20)

        self.goal = {"x": random.uniform(0 + 20, self.x_dim - 20),
                      "y": random.uniform(0 + 20, self.y_dim - 20),
                      "z": random.uniform(0 + 20, self.z_dim - 20)}

        self.path = [(self.state["x"], self.state["y"], self.state["z"])]


        return self._get_obs(), self._get_info()


    def move_agent(self, action):

        # Calculate new y position based on action
        if action == 0:
            self.state["z"] = self.state["z"] - self.alt_move
            # reward = -0.25 # Reduce score for excessive movement
        elif action == 2:
            self.state["z"] = self.state["z"] + self.alt_move
            # reward = -0.25
        else:
            pass
            #self.state["z"] = self.state["z"]  # Stay


        u, v, _ = self.FlowField3D.interpolate_flow(int(self.state["x"]), int(self.state["y"]), int(self.state["z"]))

        # Update position based on flow and timestep
        self.state["x"] = self.state["x"] + u * self.dt
        self.state["y"] = self.state["y"] + v * self.dt
        #self.state["z"] = self.state["z"] + action*self.alt_move # * self.dt

        self.state["x_vel"] = u
        self.state["y_vel"] = v
        self.state["z_vel"] = self.alt_move

        self.path.append((self.state["x"], self.state["y"], self.state["z"]))

        return 0  # Reward

    def step(self, action):
        done = False

        self.total_steps += 1

        reward = 0

        reward += self.move_agent(action)

        # Check if new position is within bounds
        if self.state["x"] < 0 or self.state["x"] > self.x_dim or\
                self.state["y"] < 0 or self.state["y"] > self.y_dim or \
                self.state["z"] < 0 or self.state["z"] > self.z_dim:
            reward += -100  # Penalize going out of bounds
            done = True

        # Check if goal has been reached
        distance_to_target = self._get_info()[
            "distance"]  # np.sqrt((self.point["x"] - self.goal["x"]) ** 2 + (self.point["y"] - self.goal["y"]) ** 2)
        if distance_to_target < 5:
            reward += 500
            print("Target Reached!", self.total_steps)
            self.target_reached = True
            done = True

            # check if episode steps length has been reached
        if self.total_steps > self.episode_length:
            reward += -100  # Penalize running out of time
            done = True

        # Observation includes point position, goal position, and flow field levels
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def _get_obs(self):
        print(self.state)

        observation = {
            'x': np.array([self.state["x"]]),
            'y': np.array([self.state["y"]]),
            'z': np.array([self.state["z"]]),
            'x_vel': np.array([self.state["x_vel"]]),
            'y_vel': np.array([self.state["y_vel"]]),
            'z_vel': np.array([self.state["z_vel"]]),
            'goal_x': np.array([self.goal["x"]]),
            'goal_y': np.array([self.goal["y"]]),
            'goal_z': np.array([self.goal["z"]])
        }

        return observation

    def _get_info(self):

        return {
            "distance": np.linalg.norm(
                np.asarray([self.state["x"], self.state["y"], self.state["z"]]) -
                np.asarray([self.goal["x"], self.goal["y"], self.goal["z"]]), ord=1),
            "target_reached": self.target_reached,
        }

    def render(self, mode='human'):

        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(18, 10))
            self.ax = self.fig.add_subplot(121, projection='3d')

            self.ax2 = self.fig.add_subplot(122, projection='3d')

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

            #Is this only done once?
            self.FlowField3D.visualize_3d_planar_flow(self.ax2, skip = 50)

            # Initialize the lines for the current frame
            self.current_state_line, = self.ax.plot([], [], [], 'r--')
            self.current_goal_line, = self.ax.plot([], [], [], 'g-')

        self.path_plot.set_data(np.array(self.path)[:, :2].T)
        self.path_plot.set_3d_properties(np.array(self.path)[:, 2])

        self.ground_track.set_data(np.array(self.path)[:, :2].T)
        self.ground_track.set_3d_properties(np.zeros(len(self.path)))



        self.scatter._offsets3d = (
        np.array([self.state["x"]]), np.array([self.state["y"]]), np.array([self.state["z"]]))

        self.scatter_goal._offsets3d = (
            np.array([self.goal["x"]]), np.array([self.goal["y"]]), np.array([self.goal["z"]]))

        # Update lines from state and goal points to the XY plane for the current frame
        self.current_state_line.set_data([self.state["x"], self.state["x"]], [self.state["y"], self.state["y"]])
        self.current_state_line.set_3d_properties([self.state["z"], 0])

        self.current_goal_line.set_data([self.goal["x"], self.goal["x"]], [self.goal["y"], self.goal["y"]])
        self.current_goal_line.set_3d_properties([self.goal["z"], 0])

        if mode == 'human':
            plt.pause(0.001)

    def close(self):
        pass


if __name__ == '__main__':
    env = FlowFieldEnv()
    obs = env.reset()

    while True:
        env.reset()
        total_reward = 0
        total_steps = 0
        for _ in range(500):
            action = 2 #env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            #print(obs, reward, done, info)
            #print(env.path)
            total_reward += reward
            total_steps += 1
            #time.sleep(1)
            env.render(mode='human')
            if done:
                break
        print("episode length", total_steps, "Total Reward", total_reward)

        # Pause for user input before starting the next episode
        input("Press Enter to continue to the next episode...")
