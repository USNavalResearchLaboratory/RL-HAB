import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
import random
import time

#from stable_baselines3.common.env_checker import check_env

from generate3dflow import FlowField

class FlowFieldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode="human"):
        self.WIDTH = 100
        self.HEIGHT = 100
        self.NUM_FLOW_LEVELS = 5
        self.dt = 0.1

        self.max_vel = 10
        self.min_vel = 1
        self.episode_length = 400

        self.render_mode = render_mode

        self.flow_field = FlowField()

        self.total_steps = 0

        self.action_space = spaces.Discrete(3)  # 0: Move down, 1: Stay, 2: Move up

        # Observation space includes continuous x and y positions and discrete flow field
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float64),
            'y': spaces.Box(low=0, high=self.HEIGHT, shape=(1,), dtype=np.float64),
            'z': spaces.Box(low=0, high=self.NUM_FLOW_LEVELS-1, shape=(1,), dtype=np.float64),
            'x_flow': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'y_flow': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'z_flow': spaces.Box(low=self.min_vel, high=self.max_vel , shape=(1,), dtype=np.float64),
            'goal_x': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float64),
            'goal_y': spaces.Box(low=0, high=self.HEIGHT, shape=(1,), dtype=np.float64),
            'goal_z': spaces.Box(low=0, high=self.NUM_FLOW_LEVELS-1, shape=(1,), dtype=np.float64)
        })


    def reset_flow(self):
        self.flow_field.generate_flow_field()

    def move_agent(self, action):
        # Calculate new x position based on horizontal flow
        new_x = self.point["x"] + self.point["x_flow"] * self.dt
        new_y = self.point["y"] + self.point["y_flow"] * self.dt
        new_z = self.point["z"] + self.point["z_flow"] * self.dt

        self.point["x"] = new_x
        self.point["y"] = new_y
        self.point["z"] = new_z

        return 0  # Reward

    def step(self, action):
        done = False

        self.total_steps += 1

        reward = 0

        reward += self.move_agent(action)

        # Observation includes point position, goal position, and flow field levels
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def _get_obs(self):
        observation = {
            'x': np.array([self.point["x"]]),
            'y': np.array([self.point["y"]]),
            'z': np.array([self.point["z"]]),
            'x_flow': np.array([self.flow_field.get_flow_at_point(self.point["x"], self.point["y"], self.point["z"])[0]]),
            'y_flow': np.array([self.flow_field.get_flow_at_point(self.point["x"], self.point["y"], self.point["z"])[1]]),
            'z_flow': np.array([self.flow_field.get_flow_at_point(self.point["x"], self.point["y"], self.point["z"])[2]]),
            'goal_x': np.array([self.goal["x"]]),
            'goal_y': np.array([self.goal["y"]]),
            'goal_z': np.array([self.goal["z"]])
        }

        return observation

    def render(self, mode='human'):
        if mode == 'human':
            # Visualize the flow field
            self.flow_field.visualize_3d()
        elif mode == 'rgb_array':
            # Return a blank image for RGB array rendering
            return np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

    def close(self):
        pass

    def horizontal_flow(self, point):
        return self.flow_field.get_flow_at_point(point["x"], point["y"], point["z"])[0]


if __name__ == '__main__':
    env = FlowFieldEnv()
    obs = env.reset()

    while True:
        env.reset()
        total_reward = 0
        total_steps = 0
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print(obs, reward, done, info)
            total_reward += reward
            total_steps += 1
            time.sleep(1)
            env.render(mode='human')
            if done:
                break
        print("episode length", total_steps, "Total Reward", total_reward)
