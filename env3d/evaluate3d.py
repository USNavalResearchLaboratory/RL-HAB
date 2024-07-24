import sys
import os
import numpy as np
sys.path.append(os.path.abspath('src'))


"""Choose which type of model to evaulate on, the static flow field or randomly generated every episode"""
#from FlowEnv3D_SK_relative import FlowFieldEnv3d
from FlowEnv3D_SK_relative_kinematics import FlowFieldEnv3d

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

### EVALUATION ### ----------------------------------------------------------------------

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)

model_name = "RL_models_3D/iconic-sponge-111/3dflow-DQN_80000000_steps"
seed = 3272669519

model_name = "RL_models_3D/easy-pond-113/3dflow-DQN_76000000_steps"
seed = None

model_name = "RL_models_3D/breezy-morning-125/3dflow-DQN_60000000_steps"
seed = None

model_name = "RL_models_km/honest-yogurt-2/DQN-km_56000000_steps"
seed = None

model_name = "RL_models/devout-dew-41/3dflow-DQN_75000000_steps"
seed = None

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

print("Loading model")


#model_name = "RL_models_3D/easy-pond-113/3dflow-DQN_80000000_steps"


env = FlowFieldEnv3d(**env_params)
model = DQN.load(model_name, env=env, )

#print(model.o)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
print ("Evaluating Model")

n_procs = 1
vec_env = model.get_env()

#'''
# Evaluate the agent with deterministic actions
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
print(f"Deterministic evaluation: mean reward = {mean_reward}, std reward = {std_reward}")

# Evaluate the agent with stochastic actions
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False)
print(f"Stochastic evaluation: mean reward = {mean_reward}, std reward = {std_reward}")
#'''



while True:
    obs = vec_env.reset()
    total_reward = 0
    total_steps = 0
    for _ in range (env_params["episode_length"]):
        action, _states = model.predict(obs, deterministic=False)
        #print(action)

        obs, rewards, dones, info = vec_env.step(action)
        #print(obs)
        total_reward += rewards
        total_steps+=1
        vec_env.render(mode='human')
        #print(info)

        if dones:
            break
    print("episode length", total_steps, "Total Reward", total_reward, "TWR", info[0]["twr"])
