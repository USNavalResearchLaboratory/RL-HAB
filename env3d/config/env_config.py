import numpy as np

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
        'render_count': 20,
        'render_skip': 50,
        'render_mode': 'human',
        'seed': np.random.randint(0, 2 ** 32),
        # A random seed needs to be defined, to generated the same random numbers across processes
    }