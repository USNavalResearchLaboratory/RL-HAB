import sys
import os
sys.path.append(os.path.abspath('src'))

from FlowEnv2D import FlowFieldEnv

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.callbacks import CheckpointCallback
#from stable_baselines3.common.callbacks import Callback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime
import time

import wandb
from wandb.integration.sb3 import WandbCallback


#Directory Initializtion

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


model_name = "DQN-2dFlow-altitude"

models_dir = "RL_models/" + model_name


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)

#Update Network Architecture
policy_kwargs = dict(net_arch=[64,64,64])


# Define hyperparameters
config = {
    "total_timesteps": int(100e6),
    'parameters': {
                'policy': "MultiInputPolicy",
                'policy_kwargs':policy_kwargs,
                'learning_rate': 1e-5,  #is 1e-6 too low?
                'exploration_fraction':.1,
                'exploration_final_eps': 0.05,
                #'learning_rate_schedule': schedule_coef,
                'batch_size': 32,
                #'n_steps': 256,
                #'ent_coef': 0.2,
                'train_freq': 4,
                'gamma': .993,
                'buffer_size': int(1e6),
                'target_update_interval': 1000,
                'stats_window_size': 1000,
            },
    "env_name": "2dflow"
    # Add other hyperparameters here
}

run = wandb.init(
    #anonymous="allow",
    project="2dflow-DQN",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

#Training Parameters
SAVE_FREQ = 500000
#TIMESTEPS = int(10e6)


# Define the checkpoint callback to save the model every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"RL_models/{run.name}",
                                          name_prefix=model_name)

# Create environment
env = FlowFieldEnv()

model = DQN(env=env,
            verbose=1,
            tensorboard_log=logdir + "/" + run.name,
            **config['parameters'])

model.learn(
    total_timesteps=config["total_timesteps"],
    # tb_log_name=model_name
    log_interval=10,
    callback=[WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"RL_models/{run.name}",
        verbose=1), checkpoint_callback],
    progress_bar=True, reset_num_timesteps=False
)

run.finish()
