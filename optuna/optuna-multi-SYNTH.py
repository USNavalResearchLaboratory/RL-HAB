import sys
import os
sys.path.append(os.path.abspath('src'))
import numpy as np
from termcolor import colored
from datetime import datetime
import pandas as pd

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch.nn as nn

import optuna
import optuna_config
from optuna.visualization.matplotlib import *
from optuna.integration.wandb import WeightsAndBiasesCallback

import wandb
from wandb.integration.sb3 import WandbCallback

# Import the Dynamics Profiles
#from era5.era5_gym import FlowFieldEnv3d
from era5.era5_gym_SYNTH import FlowFieldEnv3d_SYNTH

#import custom callbacks
from callbacks.TWRCallback import TWRCallback
from callbacks.FlowChangeCallback import FlowChangeCallback
from callbacks.TrialEvalCallback import TrialEvalCallback
from env3d.config.env_config import env_params
from era5.forecast import Forecast

import git

repo = git.Repo(search_parent_directories=True)
branch = repo.head.ref.name
hash = repo.git.rev_parse(repo.head, short=True)

def objective(trial):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "DQN_SYNTH"
    models_dir = f"{optuna_config.model_path}/{model_name}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    logdir = optuna_config.log_path
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    '''
        # Euclidian Reccomended Parameters
        # Suggest hyperparameters using Optuna
        learning_rate = trial.suggest_float('learning_rate', 2e-5, 1e-4, log=True)
        exploration_fraction = trial.suggest_float('exploration_fraction', 0.2, 0.7)
        exploration_initial_eps = trial.suggest_float('exploration_initial_eps', 0.6, 1.0)
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.05, 0.25)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1028, 2048, 4096])
        train_freq = trial.suggest_categorical('train_freq', [1, 2, 4])
        gamma = trial.suggest_float('gamma', 0.94, 0.999,  log=True)
        target_update_interval = trial.suggest_int('target_update_interval', 5000, 90000)
        buffer_size = trial.suggest_categorical('buffer_size', [int(5e5), int(1e6), int(2.5e6)]) #new one

        # Define a search space for network architecture
        num_layers = trial.suggest_int('num_layers', 4, 10)
        layer_sizes = [trial.suggest_int(f'layer_size_{i}', 32, 600) for i in range(num_layers)]
        net_arch = layer_sizes

        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=net_arch)
        '''

    # Piecewise Reccomended Parameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 7e-4, log=True)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.2, 0.85)
    exploration_initial_eps = trial.suggest_float('exploration_initial_eps', 0.25, .8)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.2)
    batch_size = trial.suggest_categorical('batch_size', [512, 1028, 2048, 4096, 8152, 16384])
    train_freq = trial.suggest_categorical('train_freq', [1, 2, 4])
    gamma = trial.suggest_float('gamma', 0.94, 0.999, log=True)
    target_update_interval = trial.suggest_int('target_update_interval', 10000, 150000)
    buffer_size = trial.suggest_categorical('buffer_size', [int(5e5), int(1e6), int(2.5e6)])

    # Define a search space for network architecture
    num_layers = trial.suggest_int('num_layers', 2, 8)
    layer_sizes = [trial.suggest_int(f'layer_size_{i}', 32, 600) for i in range(num_layers)]
    net_arch = layer_sizes

    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=net_arch)

    config = {
        "total_timesteps": optuna_config.total_timesteps,
        'parameters': {
            'policy': "MultiInputPolicy",
            'policy_kwargs': policy_kwargs,
            'learning_rate': learning_rate,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps,
            'batch_size': batch_size,
            'train_freq': train_freq,
            'gamma': gamma,
            'buffer_size': buffer_size,
            'target_update_interval': target_update_interval,
            'stats_window_size': 1000,
            'device': optuna_config.device,
        },
        "env_parameters": env_params,
        "env_name": "3dflow-km-kinematics",
        "motion_model": "Discrete", #Discrete or Kinematics, this is just a categorical note for now
        "git": branch + " - " + hash,
        "Notes": "Sep training.  Forecast threshold .5"
    }

    run = wandb.init(
        #anonymous="allow",
        project=optuna_config.project_name,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    SAVE_FREQ = int(5e6/optuna_config.n_envs)

    FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH")
    # Get month associated with Synth
    month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
    # Then process ERA5 to span the same timespan as a monthly Synthwinds File
    FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=month)

    # env = FlowFieldEnv3d(FORECAST_PRIMARY=FORECAST_PRIMARY, render_mode="human")

    env = make_vec_env(lambda: FlowFieldEnv3d_SYNTH(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH),
                       n_envs=optuna_config.n_envs)

    eval_env = DummyVecEnv([lambda: Monitor(FlowFieldEnv3d_SYNTH(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH))])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"{optuna_config.model_path}/{run.name}",
                                             name_prefix=model_name)

    #set wandb name in optuna
    trial.set_user_attr("wandb_name", run.name)

    model = DQN(env=env, verbose=1, tensorboard_log=logdir + "/" + run.name, **config['parameters'])

    try:
        model.learn(total_timesteps=config["total_timesteps"],
                    log_interval=100,
                    callback=[WandbCallback(
                        gradient_save_freq=1000,
                        model_save_path=f"{optuna_config.model_path}/{run.name}",
                        verbose=1), checkpoint_callback,
                        #TrialEvalCallback(eval_env, trial, n_eval_episodes=5, eval_freq=10000, verbose=1),
                        TWRCallback(moving_avg_length=1000, radius='twr'),
                        TWRCallback(moving_avg_length=1000, radius='twr_inner'),
                        TWRCallback(moving_avg_length=1000, radius='twr_outer'),
                        FlowChangeCallback()],
                    progress_bar=True)

        run.finish()

        #How can we add twr here?
        print(colored(f"Evaluating policy for {run.name}","cyan"))
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=50, return_episode_rewards=False)
        print(f"Mean reward for {run.name}: {mean_reward}")
        return mean_reward
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()


# Load the study from the database
study = optuna.load_study(study_name=optuna_config.project_name, storage=optuna_config.storage)

#study.optimize(objective, n_trials=n_trials, callbacks=[wandbc], n_jobs=1)
study.optimize(objective, n_trials=optuna_config.n_trials)