import sys
import os
sys.path.append(os.path.abspath('src'))
import numpy as np

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env

import wandb
from wandb.integration.sb3 import WandbCallback

#from FlowEnv3D_SK_cartesian import FlowFieldEnv3d
#from FlowEnv3D import FlowFieldEnv3d
from FlowEnv3D_SK_relative import FlowFieldEnv3d

class TargetReachedCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of the mean reward.  Tracks the moving average of the window size.
    """
    def __init__(self, moving_avg_length=1000, radius ='twr', verbose=0):
        super(TargetReachedCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]
        self.moving_avg_length = moving_avg_length
        self.target_reached_history = []
        self.radius = radius

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            infos = self.locals['infos'][0]
            #print(infos)

            self.target_reached_history.append(infos.get(self.radius))

            if len(self.target_reached_history) > self.moving_avg_length:
                self.target_reached_history.pop(0)

            moving_avg = np.mean(self.target_reached_history)
            self.logger.record('twr/' + str(self.radius), moving_avg)

        return True

class FlowChangeCallback(BaseCallback):
    """
    Custom tensorboard callback to keep track of the mean reward.  Tracks the moving average of the window size.
    """
    def __init__(self, verbose=0):
        super(FlowChangeCallback, self).__init__(verbose)
        #self.env = env  # type: Union[gym.Env, VecEnv, None]

    def _on_step(self) -> bool:
        # Check if the episode has ended
        done = self.locals['dones'][0]

        if done:
            infos = self.locals['infos'][0]

            self.logger.record('num_flow_changes', infos.get("num_flow_changes"))

        return True

#Directory Initializtion
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "DQN-km"
models_dir = "RL_models_km/" + model_name

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs_km"
if not os.path.exists(logdir):
    os.makedirs(logdir)


#Custom Network Architecture to override DQN default of 64 64
# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
policy_kwargs = dict(net_arch=[200,200,200, 200])

env_params = {
            'x_dim': 250,  # km
            'y_dim': 250,  # km
            'z_dim': 10,  # km
            'min_vel': 5 / 1000.,  # km/s
            'max_vel': 25 / 1000.,  # km/s
            'num_levels': 6,
            'dt': 60,  # seconds
            'radius': 50,  # km

            # DISCRETE
            'alt_move': 2 / 1000.,  # km/s  FOR DISCRETE

            # KINEMATICS
            'max_accel': 1.e-5,  # km/min^2
            'drag_coefficient': 0.5,

            'episode_length': 600,  # dt steps (minutes)
            'random_flow_episode_length': 1,  # how many episodes to regenerate random flow
            'decay_flow': False,
            'render_count': 1,
            'render_skip': 100,
            'render_mode': 'human',
            'seed': np.random.randint(0, 2 ** 32),
            # A random seed needs to be defined, to generated the same random numbers across processes
        }

config = {
    "total_timesteps": int(100e6),
    'hyperparameters': {
                'policy': "MultiInputPolicy",
                'policy_kwargs':policy_kwargs,
                'learning_rate': 5e-4,
                'exploration_fraction':.4,
                'exploration_initial_eps': 1,
                'exploration_final_eps': 0.1,
                'batch_size': 32,
                'train_freq': 4,
                'gamma': .99,
                'buffer_size': int(1e6),
                'target_update_interval': 10000,
                'stats_window_size': 1000,
                'device': "cuda",
            },
    "env_parameters": env_params,
    "env_name": "DQN-km",
    "motion_model": "Discrete", #Discrete or Kinematics, this is just a categorical note for now
    "NOTES": "Trying with new Trilinear Interpolation Method" #change this to lower case
}

run = wandb.init(
    #anonymous="allow",
    project="DQN-km",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

#Training Parameters

n_procs = 500
SAVE_FREQ = int(5e6/n_procs)

env = make_vec_env(lambda: FlowFieldEnv3d(**env_params), n_envs=n_procs)

# Define the checkpoint callback to save the model every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=f"RL_models_km/{run.name}",
                                          name_prefix=model_name)

model = DQN(env=env,
            verbose=1,
            tensorboard_log=logdir + "/" + run.name,
            **config['hyperparameters'],
            )

#OVerwrite
#old_model = DQN.load("RL_models_3D/faithful-oath-94/3dflow-DQN_60000000_steps", env=env, )
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
        model_save_path=f"RL_models_km/{run.name}",
        verbose=1), checkpoint_callback,
        TargetReachedCallback(moving_avg_length=1000, radius='twr'),
        TargetReachedCallback(moving_avg_length=1000, radius='twr_inner'),
        TargetReachedCallback(moving_avg_length=1000, radius='twr_outer'),
        FlowChangeCallback()],
    progress_bar=True, reset_num_timesteps=False #added this for restarting a training
)

run.finish()
