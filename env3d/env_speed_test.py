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
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.utils import set_random_seed
from line_profiler import LineProfiler
import sys
#from FlowEnv3D_SK_relative import FlowFieldEnv3d
from FlowEnv3D_SK_relative_kinematics import FlowFieldEnv3d

policy_kwargs = dict(net_arch=[200,200,200, 200])

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

config = {
    "total_timesteps": int(100e6),
    'hyperparameters': {
                'policy': "MultiInputPolicy",
                'policy_kwargs':policy_kwargs,
                'learning_rate': 5e-4,
                'exploration_fraction':.4,
                'exploration_initial_eps': 1,
                'exploration_final_eps': 0.1,
                'batch_size': 32,
                'train_freq': 4,
                'gamma': .99,
                'buffer_size': int(1e6),
                'target_update_interval': 10000,
                'stats_window_size': 1000,
                'device': "cuda",
            },
    "env_parameters": env_params,
    "env_name": "DQN-km",
    "motion_model": "Discrete", #Discrete or Kinematics, this is just a categorical note for now
    "NOTES": "" #change this to lower case
}

env = FlowFieldEnv3d(**env_params)
env.reset()

n_procs = 200
env = make_vec_env(lambda: FlowFieldEnv3d(**env_params), n_envs=n_procs)

model = DQN(env=env, verbose=1,**config['hyperparameters'])
#model = PPO(env=env, policy = "MultiInputPolicy", verbose=1)

model.learn(total_timesteps=int(10e5),log_interval=100, progress_bar=True, reset_num_timesteps=False)