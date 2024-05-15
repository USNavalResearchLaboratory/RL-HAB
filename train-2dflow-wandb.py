import sys
import os
sys.path.append(os.path.abspath('src'))

from FlowEnv2D import FlowFieldEnv

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.callbacks import CheckpointCallback
#from stable_baselines3.common.callbacks import Callback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime
import time

import wandb
from wandb.integration.sb3 import WandbCallback


#Directory Initializtion

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


model_name = "DQN-2dFlow-altitude"

models_dir = "RL_models/" + model_name


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)


'''
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
'''

#For later
policy_kwargs = dict(net_arch=[128, 128, 128])


#'''
# Define hyperparameters
config = {
    "total_timesteps": int(14e6),
    'parameters': {
                'policy': "MultiInputPolicy",
                'policy_kwargs':policy_kwargs,
                'learning_rate': 1e-4,  #is 1e-6 too low?
                'exploration_fraction':.5,
                'exploration_final_eps': 0.1,
                #'learning_rate_schedule': schedule_coef,
                'batch_size': 32,
                #'n_steps': 256,
                #'ent_coef': 0.2,
                'train_freq': 5,
                'gamma': .993,
                'buffer_size': int(5e5),
                'target_update_interval': 1000,
            },
    "env_name": "2dflow"
    # Add other hyperparameters here
}


run = wandb.init(
    #anonymous="allow",
    project="2dflow-DQN",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)



#Training Parameters
SAVE_FREQ = 100000
TIMESTEPS = int(10e6)


# Define the checkpoint callback to save the model every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"RL_models/{run.name}",
                                          name_prefix=model_name)


# Create environment
env = FlowFieldEnv()

# Instantiate the agent
#model = DQN(env = env, verbose=1,  tensorboard_log=logdir + "/" + run_id, **hyperparams)

model = DQN(env=env,
            verbose=1,
            tensorboard_log=logdir + "/" + run.name,
            **config['parameters']
        )


#model.learn(total_timesteps=TIMESTEPS,  log_interval=10, callback = checkpoint_callback, progress_bar=True, reset_num_timesteps=False)


model.learn(
    total_timesteps=config["total_timesteps"],
    # tb_log_name=model_name
    log_interval=10,
    callback=[WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"RL_models/{run.name}",
        verbose=1), checkpoint_callback],
    progress_bar=True, reset_num_timesteps=False
)

run.finish()


'''
### EVALUATION ### ----------------------------------------------------------------------

env = FlowFieldEnv(render_mode="human")

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)

print("Loading model")
model = DQN.load("RL_models/grateful-wave-22/DQN-2dFlow-altitude_7000000_steps", env=env, )

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
print ("Evaluating Model")

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
#print("mean_reward", mean_reward, "std_reward", std_reward)

# Enjoy trained agent
vec_env = model.get_env()

# Wrap the single environment in a VecEnv
#vec_env = DummyVecEnv([lambda: env])

obs = vec_env.reset()
#for i in range(1000):

while True:
    vec_env.reset()
    total_reward = 0
    total_steps = 0
    for _ in range (500):
        action, _states = model.predict(obs, deterministic=False)

        print(action)


        obs, rewards, dones, info = vec_env.step(action)
        print(obs)
        total_reward += rewards
        total_steps+=1
        vec_env.render(mode='human')
        if dones:
            break
    print("episode length", total_steps, "Total Reward", total_reward)
    
'''