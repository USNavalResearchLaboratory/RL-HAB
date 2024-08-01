import gymnasium as gym
import time

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "BipedalWalker-v3"
num_cpu = 4
n_timesteps = 10000

env = make_vec_env(env_name, n_envs=num_cpu)

model = PPO('MlpPolicy', env, verbose=0, device = "cpu")

start_time = time.time()
model.learn(n_timesteps)
total_time_multi = time.time() - start_time
print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")


single_process_model = PPO('MlpPolicy', env_name, verbose=0, device = "cpu")
start_time = time.time()
single_process_model.learn(n_timesteps)
total_time_single = time.time() - start_time


print(f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS")
print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))