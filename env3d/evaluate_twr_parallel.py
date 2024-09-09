import pandas as pd
from era5.era5_gym import FlowFieldEnv3d
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from env3d.config.env_config import env_params
from era5.forecast import Forecast

### EVALUATION ### ----------------------------------------------------------------------

model_name = "BEST_MODELS/vogons-SYNTH-piecewise/sleek-shadow-3/DQN_synth_104998950_steps"
seed = None

print("Loading model")

pres_min = env_params['pres_min']
pres_max = env_params['pres_max']
rel_dist = env_params['rel_dist']

filename = "../../../../mnt/d/FORECASTS/SYNTH-Jan-2023-SEA.nc"
FORECAST_PRIMARY = Forecast(filename)


# Wrap the environment in a DummyVecEnv
def make_env():
    return FlowFieldEnv3d(FORECAST_PRIMARY=FORECAST_PRIMARY, render_mode=None)


n_procs = 20  # Number of parallel environments
vec_env = make_vec_env(lambda: FlowFieldEnv3d(FORECAST_PRIMARY=FORECAST_PRIMARY), n_envs=n_procs)
#vec_env = DummyVecEnv([make_env for _ in range(n_procs)])

# Load the model with the vectorized environment
model = DQN.load(model_name, env=vec_env)

# Keep track of overall evaluation variables for creating heatmaps
twr_score = []
twr_inner_score = []
twr_outer_score = []
reward_score = []
forecast_score = []

NUM_EPS = int(10_000 / n_procs)
eps_length = env_params["episode_length"]

for i in range(NUM_EPS):
    obs = vec_env.reset()
    total_reward = [0] * n_procs  # Tracking total reward for each environment
    total_steps = 0

    for _ in range(eps_length):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = vec_env.step(action)

        # Sum up rewards across processes
        for j in range(n_procs):
            total_reward[j] += rewards[j]

        total_steps += 1

        # Optionally render the first environment if needed
        if infos[0]["render_mode"] == "human":
            vec_env.render(mode='human')

        if dones[0]:
            break

    # Collect the scores and append to the respective lists
    for j in range(n_procs):
        forecast_score.append(infos[j]["forecast_score"])
        twr_score.append(infos[j]["twr"])
        twr_inner_score.append(infos[j]["twr_inner"])
        twr_outer_score.append(infos[j]["twr_outer"])
        reward_score.append(total_reward[j])
        print(f"COUNT: {i}, ENV {j}")
        print(f"Forecast Score: {infos[j]['forecast_score']}")
        print(f"Total Reward: {total_reward[j]}, TWR: {infos[j]['twr']}")

# Create DataFrame to save results
df = pd.DataFrame({
    'Forecast_Score': forecast_score,
    'TWR_Inner_Score': twr_inner_score,
    'TWR_Score': twr_score,
    'TWR_Outer_Score': twr_outer_score,
    'Total_Reward': reward_score
})

df.to_csv("SYNTH-SEA-Jan-sleek-shadow-3-150m-sectors_8_parallel.csv")
print(df)
