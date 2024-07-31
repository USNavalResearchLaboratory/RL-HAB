project_name = "test_project"
model_path="RL_models_test"
log_path = "logs_test"

device = "cuda"

n_envs = 100
n_threads = 10
n_trials = 20

total_timesteps = int(1e8)

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

