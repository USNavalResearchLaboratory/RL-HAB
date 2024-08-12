import sys
import os
sys.path.append(os.path.abspath('src'))
import numpy as np

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch.nn as nn

import optuna
from optuna.visualization.matplotlib import *

import wandb
from wandb.integration.sb3 import WandbCallback

#from env3d.FlowEnv3D_SK_relative import FlowFieldEnv3d
from env3d.FlowEnv3D_SK_relative_kinematics import FlowFieldEnv3d

from callbacks.TWRCallback import TWRCallback
from callbacks.FlowChangeCallback import FlowChangeCallback

from env3d.config.env_config import env_params

import git

repo = git.Repo(search_parent_directories=True)
branch = repo.head.ref.name
hash = repo.git.rev_parse(repo.head, short=True)

def objective(trial):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "3dflow-DQN"
    models_dir = f"RL_models_3D/{model_name}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    logdir = "logs_3D"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Suggest hyperparameters using Optuna
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    exploration_fraction = trial.suggest_uniform('exploration_fraction', 0.01, 0.5)
    exploration_initial_eps = trial.suggest_uniform('exploration_initial_eps', 0.5, 1.0)
    exploration_final_eps = trial.suggest_uniform('exploration_final_eps', 0.01, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    train_freq = trial.suggest_categorical('train_freq', [1, 4, 8])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    target_update_interval = trial.suggest_int('target_update_interval', 1000, 100000)

    # Define a search space for network architecture
    num_layers = trial.suggest_int('num_layers', 1, 10)
    layer_sizes = [trial.suggest_int(f'layer_size_{i}', 32, 600) for i in range(num_layers)]
    net_arch = layer_sizes

    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=net_arch)

    config = {
        "total_timesteps": int(1e5),
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
            'buffer_size': int(1e6),
            'target_update_interval': target_update_interval,
            'stats_window_size': 1000,
            'device': "cuda",
        },
        "env_parameters": env_params,
        "env_name": "3dflow-km",
        "motion_model": "Discrete", #Discrete or Kinematics, this is just a categorical note for now
        "git": branch + " - " + hash,
        "NOTES": "Optuna tuning run"
    }

    run = wandb.init(
        #anonymous="allow",
        project="old-test",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    n_procs = 500
    SAVE_FREQ = int(5e6/n_procs)

    env = make_vec_env(lambda: FlowFieldEnv3d(**env_params), n_envs=n_procs)

    eval_env = DummyVecEnv([lambda: Monitor(FlowFieldEnv3d(**env_params))])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"RL_models_km/{run.name}",
                                             name_prefix=model_name)
    #trial_eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=5, eval_freq=10000, verbose=1)

    model = DQN(env=env, verbose=1, tensorboard_log=logdir + "/" + run.name, **config['parameters'])

    try:
        model.learn(total_timesteps=config["total_timesteps"],
                    log_interval=100,
                    callback=[WandbCallback(
                        gradient_save_freq=1000,
                        model_save_path=f"RL_models_km/{run.name}",
                        verbose=1), checkpoint_callback,
                        TWRCallback(moving_avg_length=1000, radius='twr'),
                        TWRCallback(moving_avg_length=1000, radius='twr_inner'),
                        TWRCallback(moving_avg_length=1000, radius='twr_outer'),
                        FlowChangeCallback()],
                    progress_bar=True)

        run.finish()

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=50, return_episode_rewards=False)
        return mean_reward  # Change to TWR
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()

def handle_interrupt(study, trial):
    print("\nKeyboardInterrupt detected. Stopping optimization early and returning the best result so far.")
    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")
    return study.best_params

#def make_plot(study):
#    fig = plot_optimization_history(study)
#    fig.imshow()

study = optuna.create_study(storage="sqlite:///old-test-db.sqlite3",
                            study_name="old-test",
                            direction='maximize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner())

try:
    study.optimize(objective, n_trials=100)
except KeyboardInterrupt:
    best_params = handle_interrupt(study, None)
    print(best_params)

print("Best hyperparameters: ", study.best_params)
#make_plot(study)