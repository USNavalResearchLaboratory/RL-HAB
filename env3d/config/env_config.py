import numpy as np
from era5 import config_earth

env_params = {
        'dt': config_earth.simulation['dt'],  # seconds
        'radius': 50,  # km

        'alt_min': 15000, # m
        'alt_max': 28000, # m

        # DISCRETE
        'alt_move': 2 / 1000.,  # km/s  FOR DISCRETE

        # KINEMATICS
        'max_accel': 1.e-5,  # km/min^2
        'drag_coefficient': 0.5,

        'episode_length': 600,  # dt steps (minutes)
        'random_flow_episode_length': 1,  # how many episodes to regenerate random flow
        'decay_flow': False,
        'render_count': 10,
        'render_skip': 2,
        'render_mode': 'human',
        'seed': np.random.randint(0, 2 ** 32),
        # A random seed needs to be defined, to generated the same random numbers across processes


        'rel_dist': 150000,          # m

        #These need to be mandatory pressure levels
        'pres_min': 20,              # ~27 km
        'pres_max': 200              # ~14 km
}