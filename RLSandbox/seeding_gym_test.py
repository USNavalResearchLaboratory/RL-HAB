import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import multiprocessing
from stable_baselines3 import DQN, PPO

class SimpleEnv(gym.Env):
    def __init__(self, seed=None):
        super(SimpleEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.seed(seed)

        self.total_steps = 0

    def seed(self, seed=None):
        print("updating seed", seed)
        self.np_random = np.random.default_rng(seed)
        #self.np_random = np.random.seed(seed)
        #set_random_seed(seed)

    def reset(self, seed=None):
        #if seed is not None:
        #    self.seed(seed)
        random_value = self.np_random.random()
        print("random_value", random_value)
        return np.array([random_value]), {"info": "dummy info"}


    def step(self, action):
        done = False
        if self.total_steps > 1000:
            self.total_steps = 0
            done = True

        self.total_steps +=1


        # Dummy step, not relevant for this test
        return np.array([0.0]), 0.0, done, False, {"info": "dummy info"}

def make_env(seed=None):
    def _init():
        env = SimpleEnv(seed)
        #env.seed(seed)
        return env
    return _init

def worker(seed, return_dict, idx):
    env = SimpleEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    return_dict[idx] = obs[0]

def multiproc_main():
    '''
    Demonstrate generating identical random numbers across multiple process without stable baselines3 and just
    built in multiprocessing
    '''
    num_processes = 4
    seed = 42

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []

    for i in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(seed, return_dict, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_processes):
        print(f"Random value from process {i}:", return_dict[i])

def sb3_multiproc_main():
    '''
        Demonstrate 2 different ways for generating the same random numbers across multiple processes using sb3.

        1. First way uses default make_vec_env function
        2. Second way manually declares a dummyvecenv for each process

        Uncomment which one you want to use
    '''
    n_procs = 4
    seed = 42

    # Use build in make_vec_env to create the Dummy vec enviornments
    #'''
    env = make_vec_env(lambda: SimpleEnv(seed), n_envs=n_procs, seed = seed )

    obs = env.reset()
    for i in range(n_procs):
        print(f"Random value from env {i}:", obs[i])
    #'''

    #Manually create the Dummy vec enviornments
    '''
    env_fns = [make_env(seed) for _ in range(n_procs)]
    env = DummyVecEnv(env_fns)

    obs = env.reset()
    for i in range(n_procs):
        print(f"Random value from env {i}:", obs[i])
    '''

    # Try actually learning, and it doesn't work anymore
    model = DQN("MlpPolicy", env, seed=seed, verbose=1, device="cpu")
    model.learn(total_timesteps=10000)

if __name__ == "__main__":
    #multiproc_main() #Run multiple processes of envs without using stable baselines
    sb3_multiproc_main() #Try different approaches with sb3
