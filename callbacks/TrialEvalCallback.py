from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

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