from env.RLHAB_gym_BASE import FlowFieldEnv3dBase
from env.config.env_config import env_params
from env.rendering.renderertriple import MatplotlibRendererTriple
from env.forecast_processing.forecast import Forecast_Subset
from env.forecast_processing.forecast_visualizer import ForecastVisualizer
from env.balloon import BalloonState, SimulatorState
from env.balloon import AltitudeControlCommand as command
from utils.initialize_forecast import initialize_forecasts
import random
import numpy as np
import time
from gymnasium import spaces


class FlowFieldEnv3d_DUAL(FlowFieldEnv3dBase):
    def __init__(self, FORECAST_ERA5, FORECAST_SYNTH, render_style="direction", **kwargs):
        super().__init__(**kwargs)

        self.FORECAST_ERA5 = FORECAST_ERA5
        self.FORECAST_SYNTH = FORECAST_SYNTH
        self.render_style = render_style

        self.forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)
        self.forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)

        self.forecast_subset_era5.randomize_coord(self.np_rng)
        self.forecast_subset_era5.subset_forecast(days=self.days)

        self.forecast_subset_synth.assign_coord(
            lat=self.forecast_subset_era5.lat_central,
            lon=self.forecast_subset_era5.lon_central,
            timestamp=self.forecast_subset_era5.start_time
        )
        self.forecast_subset_synth.subset_forecast(days=self.days)

        if self.render_mode == "human":
            self.vis_era5 = ForecastVisualizer(self.forecast_subset_era5)
            self.vis_era5.generate_flow_array(self.forecast_subset_era5.start_time)

            self.vis_synth = ForecastVisualizer(self.forecast_subset_synth)
            self.vis_synth.generate_flow_array(self.forecast_subset_synth.start_time)

            self.renderer = MatplotlibRendererTriple(
                Forecast_visualizer_ERA5=self.vis_era5,
                Forecast_visualizer_SYNTH=self.vis_synth,
                render_mode=self.render_mode,
                radius=self.radius,
                coordinate_system="geographic"
            )

        num_levels = len(self.forecast_subset_era5.pressure_levels)
        self.observation_space = self._build_obs_space(num_levels)
    
    def reset(self, **kwargs):
        self.forecast_scores = [5] * 4 # dummy score to trigger randomizing
        self.forecast_score = -1 # dummy score to trigger randomizing

        # For not including bad forecasts (score of 0):
        while self.forecast_score < env_params['forecast_score_threshold']:
            self.forecast_subset_era5.randomize_coord(self.np_rng)
            self.forecast_subset_era5.subset_forecast(days=self.days)

            # Then assign coord to synth winds
            self.forecast_subset_synth.assign_coord(lat=self.forecast_subset_era5.lat_central,
                                                    lon=self.forecast_subset_era5.lon_central,
                                                    timestamp=self.forecast_subset_era5.start_time)
            self.forecast_subset_synth.subset_forecast(days=self.days)


            self.forecast_scores, self.forecast_score = self.ForecastClassifier.determine_OW_Rate(self.forecast_subset_era5)


        #Reset custom metrics
        self.total_steps = 0
        self.within_target = False

        self.rogue_status = False
        self.rogue_count = 0
        self.rogue_step_trigger = None

        self.timestamp = self.forecast_subset_era5.start_time

        # Reset Balloon State to forecast subset central point.
        self.Balloon = BalloonState(lat = self.forecast_subset_era5.lat_central,
                                    lon = self.forecast_subset_era5.lon_central,
                                    x = 0,
                                    y = 0,
                                    altitude = int(random.uniform(env_params['alt_min'],env_params['alt_max']))
                                    )

        # Reset simulator state (timestamp to forecast subset start time,  counts back to 0)                            
        self.SimulatorState = SimulatorState(self.Balloon, self.forecast_subset_era5.start_time)

        # Do an artificial move to get some initial velocity, disntance, and bearing values, then reset back to initial coordinates
        self.move_agent(1)
        self.total_steps = 0 #Reset total steps for the initialization "move"

        # Synth and ERA5 don't amtter here because the variables should be identical
        self.Balloon.update(lat=self.forecast_subset_era5.lat_central, lon=self.forecast_subset_era5.lon_central, x=0, y=0, distance=0)

        # Relative wind column is still era5
        self.Balloon.rel_wind_column = self.calculate_relative_wind_column(self.forecast_subset_era5)

        if self.render_mode == "human":
            self.renderer.reset(self.goal, self.Balloon, self.SimulatorState)
            self.vis_era5.generate_flow_array(self.forecast_subset_era5.start_time)

        # Reset custom metrics
        self.twr = self.twr_inner = self.twr_outer = 0
        self.within_target = False

        return self._get_obs(), self._get_info()
    
    def get_motion_forecast(self):
        return self.forecast_subset_synth

    def get_flow_forecast(self):
        return self.forecast_subset_era5
    


def main():
    # Import Forecasts
    FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

    env = FlowFieldEnv3d_DUAL(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH, render_mode=env_params['render_mode'])

    while True:
        start_time = time.time()

        obs, info = env.reset()
        total_reward = 0
        for step in range( env_params["episode_length"]):

            # For random actions
            obs, reward, done, truncated, info = env.step(random.randint(0, 2))

            total_reward += reward

            if env_params['render_mode'] == "human":
                env.render()

            if done:
                break

        print("Total reward:", total_reward, info)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time)

if __name__ == '__main__':
    main()