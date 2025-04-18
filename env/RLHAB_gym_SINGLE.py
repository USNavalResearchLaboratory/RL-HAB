from env.RLHAB_gym_BASE import FlowFieldEnv3dBase
from env.config.env_config import env_params
from env.rendering.renderer import MatplotlibRenderer
from env.forecast_processing.forecast import Forecast_Subset
from env.forecast_processing.forecast_visualizer import ForecastVisualizer
from env.balloon import BalloonState, SimulatorState
from env.balloon import AltitudeControlCommand as command
from utils.initialize_forecast import initialize_forecasts
import random
import numpy as np
import time
from gymnasium import spaces
from typing_extensions import override

#from .RLHAB_gym_BASE import FlowFieldEnv3dBase


class FlowFieldEnv3d_SINGLE(FlowFieldEnv3dBase):
    def __init__(self, FORECAST_PRIMARY, **kwargs):
        super().__init__(**kwargs)

        # Initial randomized forecast subset from the master forecast to pass to rendering
        self.forecast_subset = Forecast_Subset(FORECAST_PRIMARY)
        self.forecast_subset.randomize_coord(self.np_rng)
        self.forecast_subset.subset_forecast(days=self.days)

        if self.render_mode=="human":
            self.Forecast_visualizer = ForecastVisualizer(self.forecast_subset)
            self.Forecast_visualizer.generate_flow_array(self.forecast_subset.start_time)  # change this for time?
            self.renderer = MatplotlibRenderer(
                                               Forecast_visualizer=self.Forecast_visualizer,
                                               render_mode=self.render_mode, radius=self.radius,  coordinate_system = "geographic")


        num_levels = len(self.forecast_subset.pressure_levels)
        self.observation_space = self._build_obs_space(num_levels)

    def reset(self, **kwargs):
        self.forecast_scores = [5] * 4 # dummy score to trigger randomizing
        self.forecast_score = -1 # dummy score to trigger randomizing

        # For not including bad forecasts (score of 0):
        while self.forecast_score < env_params['forecast_score_threshold']:
            self.forecast_subset.randomize_coord(self.np_rng)
            self.forecast_subset.subset_forecast(days=self.days)
            self.forecast_scores, self.forecast_score = self.ForecastClassifier.determine_OW_Rate(self.forecast_subset)


        #Reset custom metrics
        self.total_steps = 0
        self.within_target = False

        self.rogue_status = False
        self.rogue_count = 0
        self.rogue_step_trigger = None

        self.timestamp = self.forecast_subset.start_time

        # Reset Balloon State to forecast subset central point.
        self.Balloon = BalloonState(lat = self.forecast_subset.lat_central,
                                    lon = self.forecast_subset.lon_central,
                                    x = 0,
                                    y = 0,
                                    altitude = int(random.uniform(env_params['alt_min'],env_params['alt_max']))
                                    )

        # Reset simulator state (timestamp to forecast subset start time,  counts back to 0)                            
        self.SimulatorState = SimulatorState(self.Balloon, self.forecast_subset.start_time)

        # Do an artificial move to get some initial velocity, disntance, and bearing values, then reset back to initial coordinates
        self.move_agent(1)
        self.total_steps = 0 #Reset total steps for the initialization "move"

        self.Balloon.update(lat=self.forecast_subset.lat_central, lon=self.forecast_subset.lon_central, x=0, y=0, distance=0)
        self.Balloon.rel_wind_column = self.calculate_relative_wind_column(self.forecast_subset)

        if self.render_mode == "human":
            self.renderer.reset(self.goal, self.Balloon, self.SimulatorState)
            self.Forecast_visualizer.generate_flow_array(self.forecast_subset.start_time)

        # Reset custom metrics
        self.twr = self.twr_inner = self.twr_outer = 0
        self.within_target = False

        return self._get_obs(), self._get_info()
    
    def get_motion_forecast(self):
        return self.forecast_subset

    def get_flow_forecast(self):
        return self.forecast_subset

    '''
    def move_agent(self, action):
        # Look up flow at current 3D position before altitude change. Update Position and Flow State
        self.Balloon.lat,self.Balloon.lon,self.Balloon.x_vel,self.Balloon.y_vel, self.Balloon.x, self.Balloon.y = self.forecast_subset.getNewCoord(self.Balloon, self.SimulatorState, self.dt)

        self.Balloon.distance = np.sqrt((self.Balloon.x - self.goal["x"]) ** 2 + (self.Balloon.y - self.goal["y"]) ** 2)
        self.Balloon.rel_bearing = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                                 self.goal["x"], self.goal["y"],
                                                                 self.Balloon.x_vel, self.Balloon.y_vel)

        self.Balloon.rel_wind_column = self.calculate_relative_wind_column(self.forecast_subset)

        # Apply altitude change given action
        if action == command.UP:  # up
            # self.Balloon.z_vel = 2  # m/s
            self.Balloon.z_vel = np.random.normal(loc=env_params['ascent_rate_mean'],
                                                  scale=env_params['ascent_rate_std_dev'])
        elif action == command.DOWN:  # down
            # self.Balloon.z_vel = -3  # m/s
            self.Balloon.z_vel = -np.random.normal(loc=env_params['descent_rate_mean'],
                                                   scale=env_params['descent_rate_std_dev'])
        elif action == command.STAY:  # stay
            self.Balloon.z_vel = 0

        # Update Altitude and Last Action. Check for Altitude going out of bounds and clip.
        self.Balloon.altitude = np.clip(self.Balloon.altitude + self.Balloon.z_vel * self.dt,
                                        env_params["alt_min"],env_params["alt_max"])
        self.Balloon.last_action = action

        if self.Balloon.distance > env_params["rel_dist"]:
            if not self.rogue_status:
                self.rogue_step_trigger = self.total_steps

            self.rogue_count += 1
            self.rogue_status = True

        self.total_steps += 1

        return 0
    '''

def main():
    # Import Forecasts
    FORECAST_SYNTH, FORECAST_ERA5, forecast_subset_era5, forecast_subset_synth = initialize_forecasts()

    env = FlowFieldEnv3d_SINGLE(FORECAST_PRIMARY=FORECAST_ERA5, render_mode=env_params['render_mode'])

    while True:
        start_time = time.time()

        #env = FlowFieldEnv3d(forecast = forecast, render_mode="human")
        obs, info = env.reset()
        total_reward = 0
        for step in range(env_params["episode_length"]):


            obs, reward, done, truncated, info = env.step(random.randint(0, 2))
            total_reward += reward

            if env_params['render_mode'] == "human":
                env.render()

            if done:
                break

        print("Total reward:", total_reward, info, env_params['seed'])
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time)

if __name__ == '__main__':
    main()