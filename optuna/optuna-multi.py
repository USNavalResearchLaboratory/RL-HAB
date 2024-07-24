import sys
import os
sys.path.append(os.path.abspath('src'))
import numpy as np
from termcolor import colored

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime
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

#from FlowEnv3D_SK_cartesian import FlowFieldEnv3d
#from FlowEnv3D import FlowFieldEnv3d
#from env3d.FlowEnv3D_SK_relative import FlowFieldEnv3d
from env3d.FlowEnv3D_SK_relative_kinematics import FlowFieldEnv3d

# Add requirement for wandb core
# wandb.require("core")

class TargetReachedCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of the mean reward.  Tracks the moving average of the window size.
    """
    def __init__(self, moving_avg_length=1000, radius ='twr', verbose=0):
        super(TargetReachedCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]
        self.moving_avg_length = moving_avg_length
        self.target_reached_history = []
        self.radius = radius

        self.current_twr = 0

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            infos = self.locals['infos'][0]
            #print(infos)

            self.target_reached_history.append(infos.get(self.radius))

            if len(self.target_reached_history) > self.moving_avg_length:
                self.target_reached_history.pop(0)

            moving_avg = np.mean(self.target_reached_history)
            self.current_twr = moving_avg
            self.logger.record('twr/' + str(self.radius), moving_avg)

        return True

class FlowChangeCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of the mean reward.  Tracks the moving average of the window size.
    """
    def __init__(self, verbose=0):
        super(FlowChangeCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            infos = self.locals['infos'][0]

            self.logger.record('num_flow_changes', infos.get("num_flow_changes"))

        return True

class TrialEvalCallback(BaseCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, verbose=0):
        super(TrialEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=False)
            self.trial.report(mean_reward, self.n_calls)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
        return True




n_trials = 20

'''
wandb_kwargs = {"project": optuna_config.project_name}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
@wandbc.track_in_wandb()
'''

def objective(trial):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "3dflow-DQN"
    models_dir = f"{optuna_config.model_path}/{model_name}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    logdir = optuna_config.log_path
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Suggest hyperparameters using Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.2, 0.7)
    exploration_initial_eps = trial.suggest_float('exploration_initial_eps', 0.4, .8)
    exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.25)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1028])
    train_freq = trial.suggest_categorical('train_freq', [1, 4, 8])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    target_update_interval = trial.suggest_int('target_update_interval', 1000, 100000)
    buffer_size = trial.suggest_categorical('buffer_size', [int(5e5), int(1e6), int(2.5e6)]) #new one

    # Define a search space for network architecture
    num_layers = trial.suggest_int('num_layers', 3, 8)
    layer_sizes = [trial.suggest_int(f'layer_size_{i}', 32, 600) for i in range(num_layers)]
    net_arch = layer_sizes

    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=net_arch)

    env_params = {
        'x_dim': 250,  # km
        'y_dim': 250,  # km
        'z_dim': 10,  # km
        'min_vel': 5 / 1000.,  # km/s
        'max_vel': 25 / 1000.,  # km/s
        'num_levels': 6,
        'dt': 60,  # seconds
        'radius': 50,  # km

        # DISCRETE
        'alt_move': 2 / 1000.,  # km/s  FOR DISCRETE

        # KINEMATICS
        'max_accel': 1.e-5,  # km/min^2
        'drag_coefficient': 0.5,

        'episode_length': 600,  # dt steps (minutes)
        'random_flow_episode_length': 1,  # how many episodes to regenerate random flow
        'decay_flow': False,
        'render_count': 1,
        'render_skip': 100,
        'render_mode': 'human',
        'seed': np.random.randint(0, 2 ** 32),
        # A random seed needs to be defined, to generated the same random numbers across processes
    }

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
        "NOTES": "Optuna tuning run"
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

    env = make_vec_env(lambda: FlowFieldEnv3d(**env_params), n_envs=optuna_config.n_envs)
    eval_env = DummyVecEnv([lambda: Monitor(FlowFieldEnv3d(**env_params))])
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
                        TargetReachedCallback(moving_avg_length=1000, radius='twr'),
                        TargetReachedCallback(moving_avg_length=1000, radius='twr_inner'),
                        TargetReachedCallback(moving_avg_length=1000, radius='twr_outer'),
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
study.optimize(objective, n_trials=n_trials)