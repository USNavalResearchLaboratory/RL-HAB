
project_name = "THIS_IS_A_TEST"
model_path="RL_models_optuna"
log_path = "logs_optuna"
n_envs = 200
n_threads = 4

total_timesteps = int(5e5)

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'
