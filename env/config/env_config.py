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

        # KINEMATICS ***(Add these options in, unused for now)
        'max_accel': None,  # km/min^2
        'drag_coefficient': None,

        # RENDERING VARIABLES
        'render_count': 30,     # How many dt frames ***(is this right, or is is seconds?) to skip when rendering
        'quiver_skip': 3,       # How many quivers to skip when rendering forecast visualizations
        'alt_quiver_skip': 1,   # How many altitude quivers to skip when rendering forecast visualizations
        'render_mode': 'None', # 'human' or None to render or not during training or evaluating


        # These need to be mandatory pressure levels (https://glossarytest.ametsoc.net/wiki/Mandatory_level)
        'pres_min': 15,               # 20 for ERA5
        'pres_max': 150,              # 150 for ERA5

        # Probability Distributions for stay, ascent, and descent rates (in m/s)  Taken from EarthShab Evaluation paper ()
        'ascent_rate_mean': 1.8,
        'ascent_rate_std_dev': 0.14,
        'descent_rate_mean': 2.8,
        'descent_rate_std_dev': 0.3,
        'stay_rate_mean': 0,
        'stay_rate_std_dev': None,  #What is this value?

        'forecast_directory': "forecasts/", #"../../../../mnt/d/FORECASTS/"

        # Provided Example Forecast for running Demos
        #'era_netcdf': "ERA5-H2-2023-USA.nc",
        
        
        #'era_netcdf': "../../../cds_api/complete_era5_data/2023-ERA5-NORTH_renamed-condensed_64bit.nc",
        #'synth_netcdf': "../forecasts/SYNTH-Sep-2023-USA-UPDATED.nc",
        

        'era_netcdf': "../../../cds_api/2023-ERA5-Complete-SEA_renamed_64bit.nc",
        'synth_netcdf': "../../../FLOW2D/forecasts/SYNTH-Jan-2023-SEA-UPDATED.nc",

        #New params 
        'timewarp': 3, # None, 1, 3, 6, or 12  (for simulating faster time intervals between Synth and ERA5)

        # If using Forecast Score Decay
        'forecast_score_threshold_initial': .01,
        'forecast_score_threshold_final': .01,
        'forecast_score_threshold_decay': 0,

        # Otherwise
        'forecast_score_threshold': 0.01, # 0-1  (0.1 removes all 100% bad forecasts for navigating, winter months are typically dominated by 0 scores)

        # Evaluation Parameters
        'eval_dir': "evaluation/EVALUATION_DATA4/",
        'eval_type': "SINGLE-Synth", # DUAL or SINGLE or Baseline
        #'model_name': "RL_models_aeolus-Jan-COMPLETE/frosty-plasma-36/DQN_DUAL_COMPLETE_150000000_steps",
        #'model_name': "RL_models_aeolus-Jan-COMPLETE-bearing/playful-sound-2/DQN_DUAL_COMPLETE_150000000_steps",
        #'model_name': "RL_models_aeolus-Jan-COMPLETE-transform_fix/fast-oath-5/DQN_DUAL_COMPLETE_150000000_steps",
        #'model_name': "RL_models_aeolus-Sep-COMPLETE-tau_fix/brisk-durian-5/DQN_DUAL_COMPLETE_150000000_steps",
        'model_name': "aeolus-SEA-2023-Jul/lilac-field-3/DQN_DUAL_COMPLETE_150000000_steps",
        'eval_model': "lilac-field-3",
        'model_month': "Jan",
        'eval_month':  "Dec",
        'save_figure': False, #f or saving renderings
        'save_dir': "img/DUAL-jul_tau/"
}


def set_param(key, value):
        env_params[key] = value
        print("updating", env_params["synth_netcdf"])