import sys
import os
import numpy as np
import optuna
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch.nn as nn

# Import your custom environment
from FlowEnv3D_SK_relative_kinematics import FlowFieldEnv3d

class TargetReachedCallback(BaseCallback):
    def __init__(self, moving_avg_length=1000, verbose=0):
        super(TargetReachedCallback, self).__init__(verbose)
        self.moving_avg_length = moving_avg_length
        self.target_reached_history = []

    def _on_step(self) -> bool:
        done = self.locals['dones'][0]
        if done:
            infos = self.locals['infos'][0]
            self.target_reached_history.append(infos.get("twr"))
            if len(self.target_reached_history) > self.moving_avg_length:
                self.target_reached_history.pop(0)
            moving_avg = np.mean(self.target_reached_history)
            self.logger.record('twr', moving_avg)
        return True

class FlowChangeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(FlowChangeCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
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
        "total_timesteps": int(1e6),
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
            'device': "cpu",
        },
        "env_name": "3dflow-DQN",
        "seed": np.random.randint(0, 2 ** 32),
        "NOTES": "Optuna tuning run"
    }

    SAVE_FREQ = 250000
    n_procs = 24
    seed = config['seed']
    env = make_vec_env(lambda: FlowFieldEnv3d(seed), n_envs=n_procs, seed=seed)

    eval_env = DummyVecEnv([lambda: Monitor(FlowFieldEnv3d(seed))])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"RL_models_3D/{model_name}_{run_id}", name_prefix=model_name)
    trial_eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=5, eval_freq=10000, verbose=1)

    model = DQN(env=env, verbose=1, tensorboard_log=logdir + "/" + run_id, **config['parameters'])

    try:
        model.learn(total_timesteps=config["total_timesteps"], log_interval=100, callback=[checkpoint_callback, trial_eval_callback, TargetReachedCallback(moving_avg_length=1000), FlowChangeCallback()])
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=False)
        return mean_reward
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()

def handle_interrupt(study, trial):
    print("\nKeyboardInterrupt detected. Stopping optimization early and returning the best result so far.")
    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")
    return study.best_params

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())

try:
    study.optimize(objective, n_trials=1000)
except KeyboardInterrupt:
    best_params = handle_interrupt(study, None)
    print(best_params)

print("Best hyperparameters: ", study.best_params)
