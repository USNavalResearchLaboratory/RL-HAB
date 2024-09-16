import pandas as pd
#from FlowEnv3D_SK_relative import FlowFieldEnv3d
from era5.era5_gym_SYNTH import FlowFieldEnv3d_SYNTH
from stable_baselines3 import DQN, PPO
from env3d.config.env_config import env_params
from era5.forecast import Forecast

### EVALUATION ### ----------------------------------------------------------------------

#model_name = "BEST_MODELS/aeolus-ERA5-piecewise-extended/polished-tree-30/DQN_ERA5_300000000_steps"
#model_name = "RL_models_synth/ruby-tree-3/DQN-synth_75000000_steps"
#model_name = "BEST_MODELS/vogons-SYNTH-piecewise/sleek-shadow-3/DQN_synth_104998950_steps"
#model_name = "BEST_MODELS/aeolus-dual_USA_Jul-UPDATED/glad-lion-2/DQN_SYNTH_15000000_steps"
model_name = "BEST_MODELS/aeolus-dual_Jul-2/genial-shadow-5/DQN_SYNTH_150000000_steps"
seed = None

print("Loading model")

pres_min = env_params['pres_min']
pres_max = env_params['pres_max']
rel_dist = env_params['rel_dist']

#filename = "July-2024-SEA.nc"
#filename = "SYNTH-Jan-2023-SEA.nc"
#filename = "../../../../mnt/d/FORECASTS/SYNTH-Jan-2023-SEA.nc"
#filename = "../../../../mnt/d/FORECASTS/SYNTH-Aug-2023-USA.nc"


FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH")
# Get month associated with Synth
month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
# Then process ERA5 to span the same timespan as a monthly Synthwinds File
FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=month)

print(FORECAST_ERA5.ds_original)

env = FlowFieldEnv3d_SYNTH(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH, render_mode=None)

print(env.observation_space)


model = DQN.load(model_name, env=env, )
print(model.env.observation_space)


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

#Keep track of overall evaluation variables for creating heatmaps
twr_score = []
twr_inner_score = []
twr_outer_score = []
reward_score = []
forecast_score = []

NUM_EPS = 5000

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

    #Update scores arrays
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

df.to_csv("DUAL-Jul-on-Apr-USA-genial-shadow-piecewise.csv")
print(df)
