import sys
import os
sys.path.append(os.path.abspath('src'))

from FlowEnv2D import FlowFieldEnv

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.callbacks import CheckpointCallback
#from stable_baselines3.common.callbacks import Callback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime


#Directory Initializtion

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


model_name = "DQN_2DFlow"

models_dir = "RL_models/" + run_id


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)



# Define hyperparameters
hyperparams = {
    "policy": "MultiInputPolicy",
    #"total_timesteps": 50000,
    "learning_rate": 0.001,
    #"n_steps": 256,
    "batch_size": 32,
    'exploration_fraction': .1,
    'stats_window_size': 100,
    #"buffer_size": 10000,
    # Add other hyperparameters here
}



#Training Parameters
SAVE_FREQ = 100000
TIMESTEPS = int(10e6)


# Define the checkpoint callback to save the model every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=models_dir,
                                          name_prefix=model_name + "_" + run_id)


# Create environment
env = FlowFieldEnv()

# Instantiate the agent
model = DQN(env = env, verbose=1,  tensorboard_log=logdir + "/" + run_id, **hyperparams)


model.learn(total_timesteps=TIMESTEPS,  log_interval=10, callback = checkpoint_callback, progress_bar=True, reset_num_timesteps=False)



# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load("dqn_blimp_flow_position", env=env, )

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")