import numpy as np

env_params = {
        # TRAINING VARIABLES
        'dt': 60,               # s - How often to forward integrate simulation
        'episode_length': 1200, # dt steps (minutes) ***[I should probably change this to seconds? or is this that number times dt?]
        'radius': 50_000,       # m - Primary Station keeping radius for calculating twr
        'rel_dist': 150_000,    # m - How large to make the arena in relative distance from station keeping central coord
        'alt_min': 15_000,      # m - Minimum Altitude Bounds for training (can't go lower, HAB will stay at this altitude)
        'alt_max': 25_500,      # m - Maximum Altitude Bounds for training (can't go higher, HAB will stay at this altitude)
        'seed': 2, #np.random.randint(0, 2 ** 32), # typically ranomized, but could be set to a static seed for repeatability

        # RENDERING VARIABLES
        'render_count': 30,     # How many dt frames ***(is this right, or is is seconds?) to skip when rendering
        'quiver_skip': 3,       # How many quivers to skip when rendering forecast visualizations
        'alt_quiver_skip': 1,   # How many altitude quivers to skip when rendering forecast visualizations
        'render_mode': 'human', # 'human' or None to render or not during training or evaluating

        # These need to be mandatory pressure levels (https://glossarytest.ametsoc.net/wiki/Mandatory_level)
        'pres_min': 15,               # 20 for ERA5
        'pres_max': 150,              # 150 for ERA5

        # Probability Distributions for stay, ascent, and descent rates (in m/s)  Taken from EarthShab Evaluation paper (https://journals.ametsoc.org/view/journals/atot/41/12/JTECH-D-24-0045.1.xml)
        'ascent_rate_mean': 1.8,
        'ascent_rate_std_dev': 0.14,
        'descent_rate_mean': 2.8,
        'descent_rate_std_dev': 0.3,
        'stay_rate_mean': 0,

        'forecast_directory': "forecasts/", #"../../../../mnt/d/FORECASTS/"

        # Provided Example Forecast for running Demos
        'era_netcdf': "2023-ERA5-NORTH_renamed-condensed_64bit.nc", # Compelte ERA5.  Could replace with ERA5-H2-2023-USA.nc for Reanalysis on individual pressure levels
        'synth_netcdf': "SYNTH-Jul-2023-USA-UPDATED.nc", # Jan 2023 also included

        #New params 
        'timewarp': 3, # None, 1, 3, 6, or 12  (for simulating faster time intervals between Synth and ERA5)

        # If using Forecast Score Decay
        'forecast_score_threshold_initial': .01,
        'forecast_score_threshold_final': .01,
        'forecast_score_threshold_decay': 0,

        # Otherwise
        'forecast_score_threshold': 0.01, # 0-1  (0.1 removes all 100% bad forecasts for navigating, winter months are typically dominated by 0 scores)

        # Evaluation Parameters
        'num_evals': 1000,
        'eval_dir': "evaluation/EVALUATION_DATA/",
        'eval_type': "DUAL", # DUAL or SINGLE or Baseline for output formatting 
        'model_name': "models/Jul-Complete-USA",
        'eval_model': "Jul-Complete-USA", # model-name
        'model_month': "Jul", # training month
        'eval_month':  "Jul", # evaluation month
        # Other evaluation options
        'save_figure': False, # save renderings
        'save_dir': "img/DUAL-jul_tau/"
}

def set_param(key, value):
        env_params[key] = value
        print("updating", env_params["synth_netcdf"])