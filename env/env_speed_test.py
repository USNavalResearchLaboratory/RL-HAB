from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN
from callbacks.TWRCallback import TWRCallback
from callbacks.FlowChangeCallback import FlowChangeCallback
from callbacks.RogueCallback import RogueCallback
from callbacks.ForecastScoreDecayCallback import ForecastScoreDecayCallback
from callbacks.TimewarpCallback import TimewarpCallback
from env.config.env_config import env_params
from env.RLHAB_gym_DUAL import FlowFieldEnv3d_DUAL
#from era5.era5_gym_SINGLE import FlowFieldEnv3d_SINGLE
from utils.initialize_forecast import initialize_forecasts

import git


repo = git.Repo(search_parent_directories=True)
branch = repo.head.ref.name
hash = repo.git.rev_parse(repo.head, short=True)

policy_kwargs = dict(net_arch=[200,200,200, 200])

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
    "git": branch + " - " + hash,
    "NOTES": "" #change this to lower case
}

n_procs = 30

# Import Forecasts
FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

# Setup Env (SINGLE or Dual)
env = make_vec_env(lambda: FlowFieldEnv3d_DUAL(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH), n_envs=n_procs)
#env = make_vec_env(lambda: FlowFieldEnv3d_SINGLE(FORECAST_PRIMARY=FORECAST_ERA5), n_envs=n_procs)


model = DQN(env=env, verbose=1,**config['hyperparameters'])
#model = PPO(env=env, policy = "MultiInputPolicy", verbose=1)


model.learn(
    total_timesteps=config["total_timesteps"],
    #tb_log_name=run.name,  #added this for restarting a training
    log_interval=100,
    callback=[
        TWRCallback(moving_avg_length=1000, radius='twr'),
        TWRCallback(moving_avg_length=1000, radius='twr_inner'),
        TWRCallback(moving_avg_length=1000, radius='twr_outer'),
        FlowChangeCallback(),
        RogueCallback(),
        TimewarpCallback(),
        ForecastScoreDecayCallback(initial_percent=0.8, final_percent=0.01, decay_rate=1.0, total_timesteps=config["total_timesteps"], verbose=1)],
    progress_bar=True, reset_num_timesteps=False #added this for restarting a training
)