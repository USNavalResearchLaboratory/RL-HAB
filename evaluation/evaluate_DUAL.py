"""
An example of Evaluating a Dual (ERA5 obs, SYNTH movement)  training.  Config files need to be the same between training and evaluating
"""

import pandas as pd
from stable_baselines3 import DQN

from env.RLHAB_gym_DUAL import FlowFieldEnv3d_DUAL
from env.config.env_config import env_params
from env.forecast_processing.forecast import Forecast

model_name = "BEST_MODELS/aeolus-dual_Apr-2/effortless-blaze-23/DQN_SYNTH_150000000_steps"
seed = None

print("Loading model")

pres_min = env_params['pres_min']
pres_max = env_params['pres_max']
rel_dist = env_params['rel_dist']

FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH")
# Get month associated with Synth
month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
# Then process ERA5 to span the same timespan as a monthly Synthwinds File
FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=month)


env = FlowFieldEnv3d_DUAL(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH, render_mode=None)

model = DQN.load(model_name, env=env, )

n_procs = 1
vec_env = model.get_env()

#Keep track of evaluation variables
twr_score = []
twr_inner_score = []
twr_outer_score = []
reward_score = []
forecast_score = []

NUM_EPS = 2_000 # how many episodes to evaluate

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
    twr_rounded = round(int(info[0]["twr"] / 1200. * 100),-1)
    print("episode length", total_steps, "Total Reward", total_reward, "TWR", info[0]["twr"], twr_rounded)


    #Example For saving simulator final renderings
    #plt.savefig("pics-Oct/" + str(i) + "-Oct-FS-" + str(int(score*100)) + "-TWR-" + str(twr_rounded))
    #plt.close()


#Make Dataframe with overall scores
df = pd.DataFrame({'Forecast_Score': forecast_score,
                   'TWR_Inner_Score': twr_inner_score,
                    'TWR_Score': twr_score,
                   'TWR_Outer_Score': twr_outer_score,
                   'Total_Reward': reward_score})

eval_dir = "evaluation/EVALUATION_DATA/"
df.to_csv(eval_dir+"DUAL-Jul-on-Apr-USA-genial-shadow-piecewise_TEST.csv")
print(df)
