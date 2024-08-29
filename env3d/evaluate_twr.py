import sys
import os
import pandas as pd

import git
repo = git.Repo(search_parent_directories=True)
hash = repo.git.rev_parse(repo.head, short=True)



"""Choose which type of model to evaulate on, the static flow field or randomly generated every episode"""
#from FlowEnv3D_SK_relative import FlowFieldEnv3d
from era5.era5_gym import FlowFieldEnv3d

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from env3d.config.env_config import env_params
from era5.forecast import Forecast
import numpy as np

import matplotlib.pyplot as plt

### EVALUATION ### ----------------------------------------------------------------------

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)

def count_greater_than_zero(arr):
    return np.sum(np.array(arr) > 0)



model_name = "BEST_MODELS/aeolus-ERA5-piecewise-extended/polished-tree-30/DQN_ERA5_100000000_steps"
seed = None

print("Loading model")

pres_min = env_params['pres_min']
pres_max = env_params['pres_max']
rel_dist = env_params['rel_dist']

filename = "July-2024-SEA.nc"
FORECAST_PRIMARY = Forecast(filename)
env = FlowFieldEnv3d(FORECAST_PRIMARY=FORECAST_PRIMARY, render_mode=None)
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


twr_score = []
twr_inner_score = []
twr_outer_score = []
reward_score = []
forecast_score = []


NUM_EPS = 1000

#Visualize the Model
for i in range (0,NUM_EPS):
    print()
    print()
    obs = vec_env.reset()



    total_reward = 0
    total_steps = 0

    eps_length = env_params["episode_length"]
    for _ in range (eps_length):
        action, _states = model.predict(obs, deterministic=False)
        #print(action)

        obs, rewards, dones, info = vec_env.step(action)
        #print(obs)
        total_reward += rewards
        total_steps+=1
        #vec_env.render(mode='human')
        #print(info)

        if dones:
            break


    #5.25, 100, 2024-07-07T17:00:00.000000000
    # -6.5, 106.75, 2024-07-06T05:00:00.000000000
    #2024-07-20T11:00:00.000000000 -2.25 118.0
    #2024-07-15T22:00:00.000000000 -8.25 111.0 [0, 0, 0, 0]
    #2024-07-21T12:00:00.000000000 0.0 124.0 [0, 0, 0, 0]

    #Update scores arrays
    print("COUNT:", i)
    #print("COORD - lat:", vec_env.get_attr('dummy_lat')[0])
    #print("COORD - lon:", vec_env.get_attr('dummy_lon')[0])
    #print("COORD - time:", vec_env.get_attr('dummy_time')[0])
    #score = count_greater_than_zero(vec_env.get_attr('forecast_scores')[0])
    score = vec_env.get_attr('forecast_score')[0]
    forecast_score.append(score)
    print("Forecast Score", score)
    twr_score.append(info[0]["twr"])
    twr_inner_score.append(info[0]["twr_inner"])
    twr_outer_score.append(info[0]["twr_outer"])
    reward_score.append(total_reward[0])
    print("episode length", total_steps, "Total Reward", total_reward, "TWR", info[0]["twr"])


df = pd.DataFrame({'Forecast_Score': forecast_score,
                   'TWR_Inner_Score': twr_inner_score,
                    'TWR_Score': twr_score,
                   'TWR_Outer_Score': twr_outer_score,
                   'Total_Reward': reward_score})



fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('TWR Score for Piecewise')
ax1.scatter(df["Forecast_Score"], df["TWR_Score"])
ax2.scatter(df["Forecast_Score"], df["TWR_Inner_Score"])
ax3.scatter(df["Forecast_Score"], df["TWR_Outer_Score"])

df.to_csv("piecewise-100m-random-seed-0-1k.csv")
print(df)

plt.show()