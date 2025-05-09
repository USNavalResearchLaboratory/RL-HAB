"""Config file for optuna hyperparameter studies"""

project_name = "USA-2023-Jul"  # Name of Optuna database and Wandb project name
model_path="USA-2023-Jul"      # Where SB3 RL models will be stored
log_path = "USA-2023-Jul"           # Where SB3 log files will be stored

device = "cuda"  # CPU or cuda

n_envs = 100    # How many vectorized environment to run
n_threads = 6   # How many threaded sims to run if running in threaded mode (run_multiple_trials*.py)
n_trials = 20   # How many trials to run for optimizing hyperparameters.  If threaded this will be n_threads*n_trials

total_timesteps = int(1.5e8)    # How many timesteps per simulation.

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

