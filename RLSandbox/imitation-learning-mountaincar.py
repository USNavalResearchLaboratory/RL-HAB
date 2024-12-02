import gymnasium as gym
import numpy as np


# Define the controller for the MountainCar environment
class MountainCarController:
    def __init__(self, force=0.001, gravity=0.0025):
        self.force = force
        self.gravity = gravity
        self.max_position = 0.6
        self.min_position = -1.2
        self.max_velocity = 0.07
        self.min_velocity = -0.07

    def step(self, position, velocity, action):
        """
        Apply the transition dynamics for one step based on the action.
        """
        if action == 0:  # Accelerate left
            acceleration = -self.force
        elif action == 2:  # Accelerate right
            acceleration = self.force
        else:  # Action 1: No acceleration
            acceleration = 0

        # Compute the next velocity and position based on the current state and acceleration
        velocity_next = velocity + acceleration - np.cos(3 * position) * self.gravity
        position_next = position + velocity_next

        # Clip the position and velocity to enforce the environment limits
        position_next = np.clip(position_next, self.min_position, self.max_position)
        velocity_next = np.clip(velocity_next, self.min_velocity, self.max_velocity)

        return position_next, velocity_next


class MomentumController:
    def __init__(self, controller):
        self.controller = controller

    def get_action(self, position, velocity):
        """
        Controller that uses velocity and acceleration to determine actions.
        """
        # If velocity is negative (moving to the left) and slowing down, continue accelerating left.
        if velocity < 0 and velocity > -0.07:  # Moving left and not yet maxed out
            return 0  # Accelerate left to build more speed
        # If velocity is positive (moving to the right), continue accelerating right if speeding up.
        elif velocity > 0 and velocity < 0.07:  # Moving right and not at max speed
            return 2  # Accelerate right to climb the hill
        # If the velocity is near zero, decide whether to build momentum left or right.
        elif abs(velocity) < 0.01:
            if position < 0:  # Near the bottom of the valley
                return 0  # Accelerate left to build more speed
            else:  # We are on the right side and can now start to accelerate up
                return 2  # Accelerate right to climb the hill
        else:
            # Default case (when velocity is large and moving in the right direction)
            return 2  # Keep going right


class ExpertAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        # Simple controller logic: Push right if velocity is negative
        position, velocity = state
        if velocity < 0:
            return 2.  # Right action
        return 0.  # Left action


# Initialize the environment
env = gym.make('MountainCarContinuous-v0')
#env = gym.make('MountainCarContinuous-v0', render_mode="human")

# Create the expert agent
expert = ExpertAgent(env)


# Instantiate the controller and the bang-bang controller
controller = MountainCarController()
momentum_controller = MomentumController(controller)


# Function to collect expert demonstrations
def collect_expert_demonstrations(env, expert, num_episodes=100):

    print(env.action_space.sample())
    #sdfsdf
    demonstrations = []
    for episode in range(num_episodes):
        obs = env.reset()
        state = obs[0]
        #print(state)
        done = False
        episode_data = []
        total_reward = 0

        while not done:
            # = expert.get_action(state)  # Expert chooses action
            position, velocity = state
            action = momentum_controller.get_action(position, velocity)
            next_state, reward, done, _, info = env.step([action])  # Take action in the environment
            episode_data.append((state, action))
            state = next_state
            total_reward += reward

            #env.render()

        print(episode, total_reward )

        demonstrations.append(episode_data)

    return demonstrations


# Collect expert demonstrations
demonstrations = collect_expert_demonstrations(env, expert, num_episodes=100)
