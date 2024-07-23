project_name = "gp3-optuna-no_kinematics-new_hp_ranges"
model_path="RL_models_optuna"
log_path = "logs_optuna"
n_envs = 200
n_threads = 10

total_timesteps = int(1e8)

#Do not change
storage='sqlite:///' + project_name +'.sqlite3'

