# https://towardsdatascience.com/comparing-optimal-control-and-reinforcement-learning-using-the-cart-pole-swing-up-openai-gym-772636bc48f4

import numpy as np
import gymnasium as gym

#constants taken from here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
g= 9.8
mp = 0.1
mk = 9.8
mt = mp+mk
lp = 0.5


#### LINEARIZED DYNAMICS
# state matrix
a = g/(lp*(4.0/3 - mp/(mp+mk)))
A = np.array([[0, 1, 0, 0],
              [0, 0, a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

# input matrix
b = -1/(lp*(4.0/3 - mp/(mp+mk)))
B = np.array([[0], [1/mt], [0], [b]])

#### OPTIMAL CONTROLLER ####
R = np.eye(1, dtype=int)          # choose R (weight for input)
Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)

# get riccati solver
from scipy import linalg

# solve ricatti equation
P = linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.dot(np.linalg.inv(R),
           np.dot(B.T, P))

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left


# get environment
#env = gym.make('CartPole-v0', render_mode="human")
env = gym.make('CartPole-v1')

# Store expert demonstrations
expert_demonstrations = []
num_episodes = 100

# Generate expert demonstrations
for episode in range(num_episodes):
    obs = env.reset()[0]
    done = False
    episode_data = []
    i = 0

    while not done and i < 1000:
        # Get action from the expert controller
        action, _ = apply_state_controller(K[0], obs)  # Use your optimal controller
        episode_data.append((obs, action))

        # Take step in environment
        obs, _, done, _, _ = env.step(action)
        i+=1

    # Store the episode data
    #print(episode)
    expert_demonstrations.append(episode_data)

# Flatten the demonstrations into state-action pairs
states = []
actions = []

for episode_data in expert_demonstrations:
    for state, action in episode_data:
        states.append(state)
        actions.append(action)

states = np.array(states)
actions = np.array(actions)

print("done")