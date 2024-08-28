project_name = "aeolus-ERA5-euclidian"
model_path="RL_models_ERA5"
log_path = "logs_ERA5"

device = "cuda"

n_envs = 100
n_threads = 10
n_trials = 20

total_timesteps = int(1.5e8)

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

