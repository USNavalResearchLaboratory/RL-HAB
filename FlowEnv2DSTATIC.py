import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
import random
import time

#from stable_baselines3.common.env_checker import check_env

class FlowFieldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode="human"):
        self.WIDTH = 100
        self.HEIGHT = 100
        self.NUM_FLOW_LEVELS = 5
        self.dt = 1

        self.max_vel = 10
        self.min_vel = 1
        self.episode_length = 200

        self.render_mode = render_mode

        # Generate flow field
        self.target_reached = False
        self.reset_flow()

        self.action_space = spaces.Discrete(3)  # 0: Move down, 1: Stay, 2: Move up

        # Observation space includes continuous x and y positions and discrete flow field
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float64),
            'y': spaces.Box(low=0, high=self.HEIGHT, shape=(1,), dtype=np.float64),
            'x_flow': spaces.Box(low=-4, high=4 , shape=(1,), dtype=np.float64),
            'flow_field': spaces.Box(low=-4, high=4, shape=(self.NUM_FLOW_LEVELS,), dtype=np.float64),
            'goal_x': spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float64),
            'goal_y': spaces.Box(low=0, high=self.HEIGHT, shape=(1,), dtype=np.float64)
        })

    def reset_flow(self):
        while True:
            # Generate STATIC flow field
            static_direction = [-1,1,-1,1,-1]
            static_magnitude = [2,3,4,4,2]

            self.flow_field = np.zeros((self.NUM_FLOW_LEVELS, self.WIDTH))
            for altitude in range(self.NUM_FLOW_LEVELS):
                wind_direction = static_direction[altitude]
                wind_speed = static_magnitude[altitude]
                self.flow_field[altitude, :] = wind_direction * wind_speed

            # Count the number of right winds (1) and left winds (-1)
            right_winds = np.sum(self.flow_field > 0)
            left_winds = np.sum(self.flow_field < 0)

            # Check if there are at least 2 right winds and 2 left winds
            if right_winds >= 2 and left_winds >= 2:
                break

        #Reset visited_areas
        self.visited_areas = set()
        self.entered_area = False
        self.to_add = None

        # Interpolate between altitude levels with higher resolution
        x = np.linspace(-100, 100, self.WIDTH)
        f = interpolate.interp1d(np.arange(self.NUM_FLOW_LEVELS), self.flow_field, axis=0, kind='cubic')
        interp_intensity = self.NUM_FLOW_LEVELS
        self.interp_flow_field = f(np.linspace(0, self.NUM_FLOW_LEVELS - 1, interp_intensity))

    def altitude_reward(self):
        distance_to_goal = abs(self.point["y"] - self.goal["y"])
        if distance_to_goal <= 50:
            return 1 - distance_to_goal / 50
        else:
            return 0

    def euclidean_reward(self):
        distance_to_goal = np.linalg.norm([self.point["x"] - self.goal["x"], self.point["y"] - self.goal["y"]])
        if distance_to_goal <= 50:
            return 1 - distance_to_goal / 50
        else:
            return 0

    def euclidean_reward_exponential(self):
        distance_to_goal = np.linalg.norm([self.point["x"] - self.goal["x"], self.point["y"] - self.goal["y"]])
        if distance_to_goal <= 50:
            # Use an exponential decay function
            return np.exp(-distance_to_goal / 100)
        else:
            return 0

    def area_reward(self, resolution=5):
        reward = 0
        # Check if the agent is entering a new area
        current_area = (int(self.point["x"] // resolution), int(self.point["y"] // resolution))

        #Add the most recent newly explored area if not already in the visited map.
        if self.to_add != current_area:
            self.visited_areas.add(self.to_add)
            self.entered_area = False

        #Check if an area has already been visited an peanlize the agent.
        if current_area in self.visited_areas:
            #Only penalize the agent for the first time it enters a previously entered area.
            if not self.entered_area:
                reward -= 10  # Penalize revisiting an area
                self.entered_area = True

        #Don't add the current area until it's left the area.
        else:
            self.to_add = current_area

        return reward

    def move_agent(self, action):
        # Calculate new x position based on horizontal flow
        new_x = self.point["x"] + self.horizontal_flow(self.point) * self.dt

        reward = 0

        # Calculate new y position based on action
        if action == 0:
            new_y = max(0, self.point["y"] - 2)  # Move down
            #reward = -0.25 # Reduce score for excessive movement
        elif action == 2:
            new_y = min(self.HEIGHT - 1, self.point["y"] + 2)  # Move up
            #reward = -0.25
        else:
            new_y = self.point["y"]  # Stay


        self.point["x"] = new_x
        self.point["y"] = new_y

        self.current_flow = self.horizontal_flow(self.point)

        return reward

    def step(self, action):

        done = False

        self.total_steps += 1

        reward = 0

        reward += self.move_agent(action)

        # Append the Euclidean distance-based reward
        reward += self.euclidean_reward()

        # Penalize the agent for revisiting an area
        #reward += self.area_reward()

        # Check if new position is within bounds
        if self.point["x"] < -100 or self.point["x"] > 100 or self.point["y"] <= 0 or self.point["y"] >= self.HEIGHT-1:
            reward += -100  # Penalize going out of bounds
            done = True
            #pass

        # Check if goal has been reached
        distance_to_target =  self._get_info()["distance"]     #np.sqrt((self.point["x"] - self.goal["x"]) ** 2 + (self.point["y"] - self.goal["y"]) ** 2)
        if distance_to_target < 5:
            reward += 500
            print("Target Reached!", self.total_steps)
            self.target_reached = True
            done = True


        #check if episode steps length has been reached
        if self.total_steps > self.episode_length:
            reward += -100  # Penalize running out of time
            done = True

        # Observation includes point position, goal position, and flow field levels
        observation = self._get_obs()
        info = self._get_info()

        #print(reward)
        #print(observation)

        return observation, reward, done, False, info

    def _get_info(self):

        return {
            "distance": np.linalg.norm(
                np.asarray([self.point["x"], self.point["y"]]) - np.asarray([self.goal["x"], self.goal["y"]]), ord=1),
            "target_reached": self.target_reached,
        }

    def _get_obs(self):
        observation = {
            'x': np.array([self.point["x"]]),
            'y': np.array([self.point["y"]]),
            'x_flow': np.array([self.current_flow]),
            'flow_field': self.flow_field[:, 0],
            'goal_x': np.array([self.goal["x"]]),
            'goal_y': np.array([self.goal["y"]])
        }

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_flow()
        self.target_reached = False
        #print(self.flow_field[:, 0])

        # Reset the rendering
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            delattr(self, 'fig')
            delattr(self, 'ax')
            delattr(self, 'im')
            delattr(self, 'goal_point')
            delattr(self, 'scatter')
            delattr(self, 'canvas')

        self.total_steps = 0

        # Reset point position and include goal position in the observation
        self.point = {"x": random.uniform(-80, 80), "y": random.uniform(20, 80)}
        self.goal = {"x": random.uniform(-80, 80), "y": random.uniform(20, 80)}  # Static goal position

        self.current_flow = self.horizontal_flow(self.point)

        observation = self._get_obs()

        return observation, self._get_info()

    def render(self, mode='human'):
        # Plot flow field (only once)
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.im = self.ax.imshow(self.interp_flow_field[::-1], cmap='coolwarm', aspect='auto',
                                     extent=[-100, 100, 0, self.HEIGHT])
            plt.colorbar(self.im, ax=self.ax, label='Wind Speed', orientation='vertical')
            plt.xlabel('X')
            plt.ylabel('Altitude')
            plt.title('2D Static Flow Field with Smoothly Interpolated Altitude Levels')

            # Set y-axis limits to increase from 0 to HEIGHT
            self.ax.set_ylim(0, self.HEIGHT)

            # Plot goal position
            self.goal_point = self.ax.scatter(self.goal["x"], self.goal["y"], color='white', marker='x')

            self.canvas = self.fig.canvas

        # Update scatter plot for point mass
        if not hasattr(self, 'scatter'):
            self.scatter = self.ax.scatter(self.point["x"], self.point["y"], color='black')
        else:
            self.scatter.set_offsets([[self.point["x"], self.point["y"]]])

        if mode == 'rgb_array':
            #dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Dummy RGB image
            self.canvas.draw()
            buf = self.canvas.buffer_rgba()
            data = np.asarray(buf)
            # Convert RGBA to RGB
            rgb_data = data[:, :, :3]
            return rgb_data  # Return as a list

        elif mode == 'human':
            plt.pause(0.01)  # Pause to update plot

    def close(self):
        pass

    def horizontal_flow(self, point):
        # Calculate the altitude level based on the y position, ensuring it's within bounds
        altitude_level = np.clip(int((point["y"] / self.HEIGHT) * self.NUM_FLOW_LEVELS), 0, self.NUM_FLOW_LEVELS - 1)
        # Adjust x to be within the range [0, WIDTH-1]
        adjusted_x = np.clip(point["x"] + 100, 0, self.WIDTH - 1)
        # Interpolate the flow field at the altitude level to get the horizontal flow
        horizontal_flow = interpolate.interp1d(np.arange(self.WIDTH), self.interp_flow_field[altitude_level], kind='cubic')
        # Return the horizontal flow at the adjusted x position
        return horizontal_flow(adjusted_x)

    def is_point_valid(self, point):
        return point[0] >= -100 and point[0] <= 100 and point[1] >= 0 and point[1] <= self.HEIGHT


if __name__ == '__main__':


    # Example usage
    env = FlowFieldEnv()
    obs = env.reset()

    #env_test = check_env(env)
    #print(env_test)

    while True:
        env.reset()
        total_reward = 0
        total_steps = 0
        for _ in range(500):
            action = env.action_space.sample()
            #action = 1
            obs, reward, done, _, info = env.step(action)
            #print(obs, reward, done, info)
            total_reward += reward
            total_steps += 1
            time.sleep(.1)
            env.render(mode='human')
            if done:
                break
        print("episode length", total_steps, "Total Reward", total_reward)
