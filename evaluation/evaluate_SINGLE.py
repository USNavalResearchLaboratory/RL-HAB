"""
An example of Evaluating a SINGLE (ERA5 or SYNTH  obs+movement)  training.  Config files need to be the same between training and evaluating
"""

import pandas as pd
from env.RLHAB_gym_SINGLE import FlowFieldEnv3d_SINGLE
from stable_baselines3 import DQN
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast

model_name = "BEST_MODELS/aeolus-dual_Jul-2/genial-shadow-5/DQN_SYNTH_150000000_steps"
#env_params["episode_length"] = 1  # To override episode length, for getting quick forecast score distributions

print("Loading model")

FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH")
# Get month associated with Synth
month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
# Then process ERA5 to span the same timespan as a monthly Synthwinds File
FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=month)

FORECAST_PRIMARY = FORECAST_ERA5 # Choose Forecast_SYNTH or FORECAST_ERA5,  or manually upload a Forecast

env = FlowFieldEnv3d_SINGLE(FORECAST_PRIMARY=FORECAST_PRIMARY, render_mode="human")

model = DQN.load(model_name, env=env, )

n_procs = 1
vec_env = model.get_env()  #SB3 require vec_env for evaluating models


#Examples of simple evaluating with deterministic or stochastic actions
'''
print ("Evaluating Model")
# Evaluate the agent with deterministic actions
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
print(f"Deterministic evaluation: mean reward = {mean_reward}, std reward = {std_reward}")

# Evaluate the agent with stochastic actions
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False)
print(f"Stochastic evaluation: mean reward = {mean_reward}, std reward = {std_reward}")
'''

# Keep track of overall evaluation variables for creating heatmaps
twr_score = []
twr_inner_score = []
twr_outer_score = []
reward_score = []
forecast_score = []

NUM_EPS = 10_000  #Number of episodes to evaulate on

for i in range (0,NUM_EPS):
    obs = vec_env.reset()

    total_reward = 0
    total_steps = 0

    eps_length = env_params["episode_length"]
    for _ in range (eps_length):
        action, _states = model.predict(obs, deterministic=False)

        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards
        total_steps+=1
        #Render option
        if info[0]["render_mode"] == "human":
            vec_env.render(mode='human')

        if dones:
            break

    # Update scores arrays
    print()
    print("COUNT:", i)
    score = info[0]["forecast_score"]
    forecast_score.append(score)
    print("Forecast Score", score)
    twr_score.append(info[0]["twr"])
    twr_inner_score.append(info[0]["twr_inner"])
    twr_outer_score.append(info[0]["twr_outer"])
    reward_score.append(total_reward[0])
    print("episode length", total_steps, "Total Reward", total_reward, "TWR", info[0]["twr"])


#Make Dataframe with overall scores
df = pd.DataFrame({'Forecast_Score': forecast_score,
                   'TWR_Inner_Score': twr_inner_score,
                    'TWR_Score': twr_score,
                   'TWR_Outer_Score': twr_outer_score,
                   'Total_Reward': reward_score})

eval_dir = "evaluation/EVALUATION_DATA/"
df.to_csv(eval_dir + "SYNTH-USA-piecewise-150m-2k_evals-sectors_8-pt4.csv")
#df.to_csv("Jul-ERA5-updated.csv")
print(df)
