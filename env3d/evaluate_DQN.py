import sys
import os
import numpy as np
from utils.gitpath import get_active_branch_name
sys.path.append(os.path.abspath('src'))

branch = get_active_branch_name()
print(branch)

import subprocess
print(subprocess.check_output(["git", "describe", "--always"]).strip().decode())

import git
repo = git.Repo(search_parent_directories=True)
hash = repo.git.rev_parse(repo.head, short=True)

print(repo, repo.head.ref.name, hash)
sdfsd

"""Choose which type of model to evaulate on, the static flow field or randomly generated every episode"""
#from FlowEnv3D_SK_relative import FlowFieldEnv3d
from era5.era5_gym import FlowFieldEnv3d

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from env3d.config.env_config import env_params
from era5.forecast import Forecast

### EVALUATION ### ----------------------------------------------------------------------

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)

model_name = "RL_models_3D/iconic-sponge-111/3dflow-DQN_80000000_steps"
seed = 3272669519

model_name = "RL_models_3D/easy-pond-113/3dflow-DQN_76000000_steps"
seed = None

model_name = "RL_models_3D/breezy-morning-125/3dflow-DQN_60000000_steps"
seed = None

model_name = "RL_models_km/honest-yogurt-2/DQN-km_56000000_steps"
seed = None

model_name = "RL_models/devout-dew-41/3dflow-DQN_75000000_steps"
seed = None

model_name = "RL_models/bright-pyramid-103/3dflow-DQN_95000000_steps"
seed = None

model_name = "RL_models_era5/tough-cloud-1/DQN-ERA5_30000000_steps"
seed = None

model_name = "RL_models_era5/royal-water-2/DQN-ERA5_15000000_steps"
seed = None

print("Loading model")

forecast = Forecast(env_params['rel_dist'], env_params['pres_min'], env_params['pres_max'])
env = FlowFieldEnv3d(forecast=forecast, render_mode="human")
model = DQN.load(model_name, env=env, )

#print(model.o)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.


n_procs = 1
vec_env = model.get_env()

'''
print ("Evaluating Model")
# Evaluate the agent with deterministic actions
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
print(f"Deterministic evaluation: mean reward = {mean_reward}, std reward = {std_reward}")

# Evaluate the agent with stochastic actions
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False)
print(f"Stochastic evaluation: mean reward = {mean_reward}, std reward = {std_reward}")
'''

#Visualize the Model
while True:
    obs = vec_env.reset()
    total_reward = 0
    total_steps = 0
    for _ in range (env_params["episode_length"]):
        action, _states = model.predict(obs, deterministic=False)
        #print(action)

        obs, rewards, dones, info = vec_env.step(action)
        #print(obs)
        total_reward += rewards
        total_steps+=1
        vec_env.render(mode='human')
        #print(info)

        if dones:
            break
    print("episode length", total_steps, "Total Reward", total_reward, "TWR", info[0]["twr"])
