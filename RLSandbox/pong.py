from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, PPO, DQN
import os
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from datetime import datetime




#Directory Initializtion

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


model_name = "A2C_pong"

models_dir = "RL_models/" + run_id


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)


SAVE_FREQ = 50000
# Define the checkpoint callback to save the model every 1000 steps
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=models_dir,
                                          name_prefix=model_name + "_" + run_id)




# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)

#model = A2C("CnnPolicy", vec_env, verbose=1)
#model = A2C("CnnPolicy", vec_env, verbose=1, tensorboard_log=logdir + "/" + run_id)

'''
model = PPO(policy = "CnnPolicy",
            env = vec_env,
            batch_size = 256,
            clip_range = 0.1,
            ent_coef = 0.01,
            gae_lambda = 0.9,
            gamma = 0.99,
            learning_rate = 2.5e-4,
            max_grad_norm = 0.5,
            n_epochs = 4,
            n_steps = 128,
            vf_coef = 0.5,
            tensorboard_log = logdir + "/" + run_id,
            verbose=1,
            )
'''
'''
model = PPO(policy = "CnnPolicy",
            env = vec_env,
            tensorboard_log = logdir + "/" + run_id,
            verbose=1,
            )
'''

model = PPO(ent_coef = 0.01,
             policy = 'CnnPolicy',
            env = vec_env,
             #normalize = False,
            tensorboard_log=logdir + "/" + run_id,
            verbose=1,
            )


total_timesteps = int(10e6)


#model.learn(total_timesteps=25_000, verbose=1,  tensorboard_log=logdir)

model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True, reset_num_timesteps=False)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")