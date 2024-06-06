import sys
import os
sys.path.append(os.path.abspath('src'))

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env

import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
from collections import deque

from FlowEnv3Dstationkeeping import FlowFieldEnv3d


class TargetReachedCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of the mean reward.  Tracks the moving average of the window size.
    """
    def __init__(self, moving_avg_length=1000, verbose=0):
        super(TargetReachedCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]
        self.moving_avg_length = moving_avg_length
        self.target_reached_history = []

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            infos = self.locals['infos'][0]
            #print(infos)

            self.target_reached_history.append(infos.get("twr"))

            if len(self.target_reached_history) > self.moving_avg_length:
                self.target_reached_history.pop(0)

            moving_avg = np.mean(self.target_reached_history)
            self.logger.record('twr', moving_avg)

        return True

#Directory Initializtion
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "3dflow-DQN"
models_dir = "RL_models_3D/" + model_name

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs_3D"
if not os.path.exists(logdir):
    os.makedirs(logdir)


#Custom Network Architecture to override DQN default of 64 64
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
policy_kwargs = dict(net_arch=[64, 64, 64, 64])

# Define hyperparameters
config = {
    "total_timesteps": int(100e6),
    'parameters': {
                'policy': "MultiInputPolicy",
                'policy_kwargs':policy_kwargs,
                'learning_rate': 1e-5,
                'exploration_fraction':.25,
                'exploration_initial_eps': .7,
                'exploration_final_eps': 0.1,
                'batch_size': 32,
                'train_freq': 4,
                'gamma': .99,
                'buffer_size': int(1e6),
                'target_update_interval': 10000,
                'stats_window_size': 1000,
                'device': "cpu",

            },
    "env_name": "3dflow-DQN",
    "NOTES": "Random Flow every 1000 episodes.  4 layer 64 node network" #change this to lower case

    # Add other hyperparameters here
}

run = wandb.init(
    #anonymous="allow",
    project="3dflow-DQN",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

#Training Parameters
SAVE_FREQ = 250000  #with num of procs = 16,  this will be every 2 mil steps
#TIMESTEPS = int(10e6)

n_procs = 16
#SAVE_FREQ = 500000/16

env = make_vec_env(FlowFieldEnv3d, n_envs=n_procs)


# Define the checkpoint callback to save the model every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"RL_models_3D/{run.name}",
                                          name_prefix=model_name)

# Create environment
#env = FlowFieldEnv()
#env = Monitor(env)



model = DQN(env=env,
            verbose=1,
            tensorboard_log=logdir + "/" + run.name,
            **config['parameters'],
            )

#OVerwrite
#old_model = DQN.load("RL_models_3D/super-salad-20/3dflow-DQN_44000000_steps", env=env, )
# Extract the policy weights
#policy_weights = old_model.policy.state_dict()
# Load the policy weights into the new model
#model.policy.load_state_dict(policy_weights)


#################
# MAKE SURE TO CHANGE THIS STUFF BACK! reset_timesteps = false
#########################

model.learn(
    total_timesteps=config["total_timesteps"],
    #tb_log_name=run.name,  #added this for restarting a training
    log_interval=100,
    callback=[WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"RL_models_3D/{run.name}",
        verbose=1), checkpoint_callback, TargetReachedCallback(moving_avg_length=1000)],
    progress_bar=True, reset_num_timesteps=False #added this for restarting a training
)

run.finish()
