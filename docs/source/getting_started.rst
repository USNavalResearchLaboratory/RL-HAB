Getting Started
================

Tutorial goes here...

Downloading ERA5 forecasts
___________________________________________

Generating Synthetic Forecasts
___________________________________________

Running a simulation in Gym
___________________________________________

The RL-HAB simulation environment is built on the `Gym API <https://www.gymlibrary.dev/index.html>`_. This API is a standard API for most reinforcement learning applications, and can be easily coupled with out-of-the-box RL implementations such as `StableBaselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_.
There are two types of custom Gym environments to choose from: `RLHAB_gym_SINGLE.py` and `RLHAB_gym_DUAL.py`.
- What are the two types of custom Gym envrionments and what do each mean
- Requirements for every environment (ERA5 + Synth)
- Modify the env_config.py file
    - What do each of the variables here mean? Which ones are open for modification? - Should this link to the API?
    - Dynamics noise
- Example call of the environment with random actions
Training an agent with DQN
___________________________________________
- StableBaselines3 DQN explanation
    - Simple training example usage
- Optuna explanation + usage
    - Optuna dashboard
- Weights and Biases explanation + usage
    - Screenshot of W&Biases
    
Evaluating an Agent
___________________________________________