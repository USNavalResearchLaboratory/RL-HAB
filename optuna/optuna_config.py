project_name = "aeolus-synth-piecewise"
model_path="RL_models_synth"
log_path = "logs_synth"

device = "cuda"

n_envs = 100
n_threads = 5
n_trials = 20

total_timesteps = int(1.5e8)

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

