"""
An example of Evaluating a Dual (ERA5 obs, SYNTH movement)  training.  Config files need to be the same between training and evaluating
"""

import pandas as pd
from stable_baselines3 import DQN
import argparse

from env.RLHAB_gym_DUAL import FlowFieldEnv3d_DUAL
from env.config.env_config import env_params
from utils.initialize_forecast import initialize_forecasts
from env.forecast_processing.forecast import Forecast, Forecast_Subset


# Load default parameters from EnvParams
default_eval_month = env_params["eval_month"]
default_era_netcdf = env_params["era_netcdf"]
default_synth_netcdf = env_params["synth_netcdf"]


# Set up argument parser
parser = argparse.ArgumentParser(description="Run evaluation with optional overrides.")
parser.add_argument(
    "--month",
    default=default_eval_month,
    help=f"Evaluation month (default: {default_eval_month})"
)
parser.add_argument(
    "--era_netcdf",
    default=default_era_netcdf,
    help=f"Path to ERA netCDF file (default: {default_era_netcdf})"
)
parser.add_argument(
    "--synth_netcdf",
    default=default_synth_netcdf,
    help=f"Path to Synth netCDF file (default: {default_synth_netcdf})"
)

# Parse command-line arguments
args = parser.parse_args()

# Use args.<param> in the script
print(f"Running evaluation with:")
print(f"  Month: {args.month}")
print(f"  ERA netCDF: {args.era_netcdf}")
print(f"  Synth netCDF: {args.synth_netcdf}")

env_params["eval_month"] = args.month
env_params["era_netcdf"] = args.era_netcdf
env_params["synth_netcdf"] = args.synth_netcdf

import os

#model_name = "BEST_MODELS/aeolus-dual_Jul-2/genial-shadow-5/DQN_SYNTH_150000000_steps"
#model_name = "BEST_MODELS/aeolus-dual_Jan-2/serene-meadow-14/DQN_SYNTH_150000000_steps"
#model_name = "BEST_MODELS/aeolus-dual_Jul-rogue-notimewarp/wild-lion-24/DQN_DUAL_ROGUE_150000000_steps"
model_name = env_params["model_name"] #BEST_MODELS/aeolus-dual_Jul-custom-hps/silvery-jazz-1/DQN_DUAL_ROGUE_150000000_steps"

print("Loading model")

pres_min = env_params['pres_min']
pres_max = env_params['pres_max']
rel_dist = env_params['rel_dist']


#overrided initialize_forecasts

# Import Forecasts
FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH", timewarp=env_params['timewarp'])
# Get month associated with Synth
synth_month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
# Then process ERA5 to span the same timespan as a monthly Synthwinds File
FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=synth_month, timewarp=env_params['timewarp'])

forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)
forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)


env = FlowFieldEnv3d_DUAL(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH, render_mode=env_params["render_mode"])

model = DQN.load(model_name, env=env, )

#Disable Exploration in Evaulation?
#model.policy.set_training_mode(False)

print("Training timesteps:", model.num_timesteps)

replay_buffer = model.replay_buffer
print(f"Replay buffer size: {replay_buffer.size()}")
#print(model.__dict__)
#print(model.__dict__['_last_original_obs']["flow_field"].shape)
#sdfsdf

#sdfsdf

n_procs = 1
vec_env = model.get_env()

#Keep track of evaluation variables
twr_score = []
twr_inner_score = []
twr_outer_score = []
reward_score = []
forecast_score = []

rogue = []
rogue_percent = []

lats = []
lons = []
timestamps = []
altitudes = []

NUM_EPS = 1000 # how many episodes to evaluate

for i in range (0,NUM_EPS):

    rogue_status = 0
    rogue_cumulative = 0

    obs = vec_env.reset()

    total_reward = 0
    total_steps = 0

    eps_length = env_params["episode_length"]
    for _ in range (eps_length):
        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards
        total_steps+=1
        #Render option
        if info[0]["render_mode"] == "human":
            vec_env.render(mode='human')

        if info[0]["distance"] > env_params["rel_dist"]:
            rogue_status = 1
            rogue_cumulative += 1

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

    rogue.append(rogue_status)
    rogue_percent.append(rogue_cumulative / (total_steps*1.))

    print("episode length", total_steps, "Total Reward", total_reward, "TWR", info[0]["twr"], "Rogue", rogue_status, "Rogue Percent", rogue_cumulative / (total_steps*1.)  )
    print("Coord", env.forecast_subset_era5.start_time, env.forecast_subset_era5.lat_central,env.forecast_subset_era5.lon_central, env.Balloon.altitude)
        
    timestamps.append(env.forecast_subset_era5.start_time)
    lats.append(env.forecast_subset_era5.lat_central)
    lons.append(env.forecast_subset_era5.lon_central)
    altitudes.append(env.Balloon.altitude)

    #Example For saving simulator final renderings
    #plt.savefig("pics-Oct/" + str(i) + "-Oct-FS-" + str(int(score*100)) + "-TWR-" + str(twr_rounded))
    #plt.close()


#Make Dataframe with overall scores
df = pd.DataFrame({'Forecast_Score': forecast_score,
                   'TWR_Inner_Score': twr_inner_score,
                    'TWR_Score': twr_score,
                   'TWR_Outer_Score': twr_outer_score,
                   'Total_Reward': reward_score,
                    'rogue': rogue,
                   'rogue_status': rogue_percent,
                       'timestamp': timestamps,
                       'lon': lons,
                       'lat': lats,
                       'altitude': altitudes}
                       )

eval_dir = env_params["eval_dir"]
full_dir = eval_dir + '/' + env_params["eval_type"] + "_" + env_params["eval_model"] +  "_" + env_params["model_month"] + "-deterministic/"
#env_params["eval_month"] = "Jan"

if not os.path.exists(full_dir):
    os.makedirs(full_dir)


df.to_csv(full_dir + env_params["eval_type"] + "-" + env_params["model_month"] + "-on-" + env_params["eval_month"] + "-" + env_params["eval_model"] + "-deterministic.csv")
print(df)
