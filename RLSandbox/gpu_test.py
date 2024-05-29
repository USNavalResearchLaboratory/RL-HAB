import gymnasium as gym
import time
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

t1 = time.time()
model = PPO("MlpPolicy", env, verbose=0, device="cuda")
model.learn(total_timesteps=10_000)
print(f"Time with cuda : {time.time()-t1:.2f}s")

t1 = time.time()
model = PPO("MlpPolicy", env, verbose=0, device="cpu")
model.learn(total_timesteps=10_000)
print(f"Time with cpu : {time.time()-t1:.2f}s")

env.close()