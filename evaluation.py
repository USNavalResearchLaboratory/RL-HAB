import sys
import os
sys.path.append(os.path.abspath('src'))

from FlowEnv2D import FlowFieldEnv

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

### EVALUATION ### ----------------------------------------------------------------------

env = FlowFieldEnv(render_mode="human")

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)

print("Loading model")
model = DQN.load("RL_models/confused-dust-31/DQN-2dFlow-altitude_1500000_steps", env=env, )
model = DQN.load("RL_models/vocal-plant-28/DQN-2dFlow-altitude_7000000_steps", env=env, )

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
