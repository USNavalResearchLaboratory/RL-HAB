import numpy as np
import xarray as xr

env_params = {
        'dt': 60,  # seconds
        'radius': 50_000,  # km

        'alt_min': 15_000, # m
        'alt_max': 25_500, # m

        # DISCRETE
        'alt_move': 2 / 1000.,  # km/s  FOR DISCRETE

        # KINEMATICS
        'max_accel': 1.e-5,  # km/min^2
        'drag_coefficient': 0.5,

        'episode_length': 1200,  # dt steps (minutes)
        'random_flow_episode_length': 1,  # how many episodes to regenerate random flow
        'decay_flow': False,
        'render_count': 30,
        'render_skip': 2,
        'render_mode': 'human',
        'seed': np.random.randint(0, 2 ** 32),
        # A random seed needs to be defined, to generated the same random numbers across processes


        'rel_dist': 150_000,          # m

        #These need to be mandatory pressure levels
        'pres_min': 15,              # 20 for ERA5
        'pres_max': 150,              # 150 for ERA5

# Probability Distributions for ascent and descent (in m/s)
        'ascent_rate_mean': 1.8,
        'ascent_rate_std_dev': 0.14,
        'descent_rate_mean': 2.8,
        'descent_rate_std_dev': 0.3,

        'forecast_directory': "forecasts/", #"../../../../mnt/d/FORECASTS/ERA5-H2-2023-USA.nc"

        # Provided Example Forecast for running Demos
        'era_netcdf': "ERA5-H2-2023-USA.nc",
        'synth_netcdf': "SYNTH-Jul-2023-USA-UPDATED.nc",

        'forecast_score_threshold': 0.01
}