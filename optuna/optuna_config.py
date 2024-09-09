project_name = "aeolus-era5_obs_synth_movement"
model_path="RL_models_era5_obs_synth_movement"
log_path = "logs_era5_obs_synth_movement"

device = "cuda"

n_envs = 100
n_threads = 5
n_trials = 20

total_timesteps = int(1.5e8)

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

