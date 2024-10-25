"""Config file for optuna hyperparemter studies"""

project_name = "RL_models_era5_obs_synth_movement"  # Name of Optuna database and Wandb project name
model_path="RL_models_era5_obs_synth_movement"      # Where SB3 RL models will be stored
log_path = "logs_era5_obs_synth_movement"           # Where SB3 log files will be stored

device = "cuda"  # CPU or cuda

n_envs = 100    # How many vectorized environemnt to run
n_threads = 5   # How many threaded sims to run if running in threaded mode (run_multiple_trials*.py)
n_trials = 20   # How many trials to run for optimizing hyperparemters.  If threaded this will be n_threads*n_trials

total_timesteps = int(1.5e8)    # How many timesteps per simulation.

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

