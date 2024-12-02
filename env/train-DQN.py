import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env

import wandb
from wandb.integration.sb3 import WandbCallback
from env.forecast_processing.forecast import Forecast

#from FlowEnv3D_SK_relative import FlowFieldEnv3d
#from FlowEnv3D_SK_relative_kinematics import FlowFieldEnv3d
from env.RLHAB_gym_SINGLE import FlowFieldEnv3d_SINGLE
from env.RLHAB_gym_DUAL import FlowFieldEnv3d_DUAL

from callbacks.TWRCallback import TWRCallback
from callbacks.FlowChangeCallback import FlowChangeCallback
from callbacks.RogueCallback import RogueCallback
from callbacks.ForecastScoreDecayCallback import ForecastScoreDecayCallback
from callbacks.TimewarpCallback import TimewarpCallback
from env.config.env_config import env_params
from utils.initialize_forecast import initialize_forecasts

import git

repo = git.Repo(search_parent_directories=True)
branch = repo.head.ref.name
hash = repo.git.rev_parse(repo.head, short=True)

# Directory Initialization
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "DQN-DUAL-TEST"
models_dir = "RL_models_DUAL-TEST/" + model_name

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs_DUAL-TEST"
if not os.path.exists(logdir):
    os.makedirs(logdir)


# Custom Network Architecture to override DQN default of 64 64
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
policy_kwargs = dict(net_arch=[75, 275, 500, 500, 425, 500, 125, 250])

config = {
    "total_timesteps": int(150e6),
    'hyperparameters': {
                'policy': "MultiInputPolicy",
                'policy_kwargs':policy_kwargs,
                'learning_rate': .00003,
                'exploration_fraction': 0.25,
                'exploration_initial_eps': 0.75,
                'exploration_final_eps': 0.1,
                'batch_size': 512,
                'train_freq': 1,
                'gamma': .95,
                'buffer_size': int(0.5e6),
                'target_update_interval': 125_000,
                'stats_window_size': 1000,
                'device': "cuda",
            },
    "env_parameters": env_params,
    "env_name": "DQN-DUAL",
    "motion_model": "Discrete", #Discrete or Kinematics, this is just a categorical note for now
    "git": branch + " - " + hash,
    "NOTES": "Testing Rogue functionality. Using January with serene meadow hyperparams"
}

run = wandb.init(
    #anonymous="allow",
    project="DQN-DUAL-ROGUE-TEST",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

#Training Parameters

n_procs = 100
SAVE_FREQ = int(5e6/n_procs)

# Import Forecasts
FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

env = make_vec_env(lambda: FlowFieldEnv3d_DUAL(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH), n_envs=n_procs)

# Define the checkpoint callback to save the model every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"RL_models_synth/{run.name}",
                                          name_prefix=model_name)

model = DQN(env=env,
            verbose=1,
            tensorboard_log=logdir + "/" + run.name,
            **config['hyperparameters'],
            )

#OVerwrite
#old_model = DQN.load("RL_models_3D/faithful-oath-94/3dflow-DQN_60000000_steps", env=env, )
# Extract the policy weights
#policy_weights = old_model.policy.state_dict()
# Load the policy weights into the new model
#model.policy.load_state_dict(policy_weights)


#################
# MAKE SURE TO CHANGE THIS STUFF BACK! reset_timesteps = false
#########################

model.learn(
    total_timesteps=config["total_timesteps"],
    #tb_log_name=run.name,  #added this for restarting a training
    log_interval=100,
    callback=[WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"RL_models_synth/{run.name}",
        verbose=1), checkpoint_callback,
        TWRCallback(moving_avg_length=1000, radius='twr'),
        TWRCallback(moving_avg_length=1000, radius='twr_inner'),
        TWRCallback(moving_avg_length=1000, radius='twr_outer'),
        RogueCallback(),
        FlowChangeCallback(),
        TimewarpCallback(),
        ForecastScoreDecayCallback(initial_percent=0.8, final_percent=0.01, decay_rate=1.0,
                                   total_timesteps=config["total_timesteps"])],
    progress_bar=True, reset_num_timesteps=False #added this for restarting a training
)

run.finish()
