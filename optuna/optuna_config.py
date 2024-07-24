project_name = "gp3-optuna-kinematics"
model_path="RL_models_optuna"
log_path = "logs_optuna"

device = "cuda:1"

n_envs = 200
n_threads = 4

total_timesteps = int(1e8)

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

