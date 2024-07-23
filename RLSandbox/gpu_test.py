import gymnasium as gym
import time
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

#'''
n_procs = 4
#env = gym.make("CartPole-v1")
env = make_vec_env("LunarLander-v2", n_envs=n_procs)

'''
# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env("PongNoFrameskip-v4", n_envs=n_procs, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=n_procs)
'''

#policy = "CnnPolicy" #use for pong
policy = "MlpPolicy" #use for pong #use for Mlp
time_steps = 10_000_000

'''
print("A2C attempt")
t = time.time()
model = A2C(policy, env, verbose=0, device="cpu")
print(f"cuda load time : {time.time()-t:.2f}s")
t1 = time.time()
model.learn(total_timesteps=time_steps, progress_bar=True)
print(f"Time with cuda : {time.time()-t1:.2f}s")

t2 = time.time()
model = A2C(policy, env, verbose=0, device="cpu")
model.learn(total_timesteps=time_steps, progress_bar=True)
print(f"Time with cpu : {time.time()-t2:.2f}s")
print()
'''

print("DQN attempt")
t = time.time()
model = DQN(policy, env, verbose=0, device="cuda")
print(f"cuda load time : {time.time()-t:.2f}s")
t1 = time.time()
model.learn(total_timesteps=time_steps, progress_bar=True)
print(f"Time with cuda : {time.time()-t1:.2f}s")

t2 = time.time()
model = DQN(policy, env, verbose=0, device="cpu")
model.learn(total_timesteps=time_steps, progress_bar=True)
print(f"Time with cpu : {time.time()-t2:.2f}s")
print()

print("PPO attempt")
t = time.time()
model = PPO(policy, env, verbose=0, device="cuda")
print(f"cuda load time : {time.time()-t:.2f}s")
t1 = time.time()
model.learn(total_timesteps=time_steps, progress_bar=True)
print(f"Time with cuda : {time.time()-t1:.2f}s")

t2 = time.time()
model = PPO(policy, env, verbose=0, device="cpu")
model.learn(total_timesteps=time_steps, progress_bar=True)
print(f"Time with cpu : {time.time()-t2:.2f}s")
print()



env.close()
#'''
