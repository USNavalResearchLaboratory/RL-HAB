import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import time
from pynput import keyboard
import math
from env.rendering.renderertriple import MatplotlibRendererTriple
from env.config.env_config import env_params
from env.forecast_processing.forecast_visualizer import ForecastVisualizer
from env.balloon import BalloonState, SimulatorState
from env.balloon import AltitudeControlCommand as command
from env.forecast_processing.forecast import Forecast, Forecast_Subset
from env.forecast_processing.ForecastClassifier import ForecastClassifier

from utils.common import convert_range
import pandas as pd

np.set_printoptions(suppress=True, precision=3)

class FlowFieldEnv3d_DUAL(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    # UPDATE: Now the enviornment takes in parameters we can keep track of.

    def __init__(self, FORECAST_ERA5, FORECAST_SYNTH, days = 1, seed=None, render_mode=None, render_style = "direction" ):
        super(FlowFieldEnv3d_DUAL, self).__init__()

        self.days = days

        self.FORECAST_SYNTH = FORECAST_SYNTH
        self.FORECAST_ERA5 = FORECAST_ERA5

        self.render_style = render_style


        self.ForecastClassifier = ForecastClassifier()
        self.dt = env_params['dt']
        self.render_mode = render_mode

        self.seed(seed)

        self.radius = env_params['radius'] # station keeping radius
        self.radius_inner = self.radius *.5
        self.radius_outer = self.radius * 1.5

        #Goal never changes since we are using relative distance. Spawn point may change though.
        self.goal = {"x": 0,
                     "y": 0}

        # Initial randomized forecast subset from the master forecast to pass to rendering
        self.forecast_subset_era5 = Forecast_Subset(FORECAST_ERA5)
        self.forecast_subset_era5.randomize_coord()
        self.forecast_subset_era5.subset_forecast(days=self.days)


        #Then assign coord to synth winds
        self.forecast_subset_synth = Forecast_Subset(FORECAST_SYNTH)
        self.forecast_subset_synth.assign_coord(lat = self.forecast_subset_era5.lat_central,
                                                lon = self.forecast_subset_era5.lon_central,
                                                timestamp= self.forecast_subset_era5.start_time)
        self.forecast_subset_synth.subset_forecast(days=self.days)


        self.forecast_scores = [5, 5, 5, 5] # dummy score to trigger randomizing


        if self.render_mode=="human":
            self.Forecast_visualizer = ForecastVisualizer(self.forecast_subset_era5)
            self.Forecast_visualizer.generate_flow_array(self.forecast_subset_era5.start_time)  # change this for time?

            self.Forecast_visualizer_synth = ForecastVisualizer(self.forecast_subset_synth)
            self.Forecast_visualizer_synth.generate_flow_array(self.forecast_subset_synth.start_time)  # change this for time?

            self.renderer = MatplotlibRendererTriple(
                                               Forecast_visualizer_ERA5=self.Forecast_visualizer,
                                               Forecast_visualizer_SYNTH=self.Forecast_visualizer_synth,
                                               render_mode=self.render_mode, radius=self.radius,  coordinate_system = "geographic")

        self.action_space = spaces.Discrete(3)  # 0: Move down, 1: Stay, 2: Move up


        min_vel = 0  # m/s
        max_vel = 50 # m/s
        num_levels = len(self.forecast_subset_era5.pressure_levels) # determined from pressure levels


        # These number ranges are technically wrong.  Velocity could be 0?  Altitude can currently go out of bounds.
        self.observation_space = spaces.Dict({
            'altitude': spaces.Box(low=env_params['alt_min'], high=env_params['alt_max'], shape=(1,), dtype=np.float64),
            'distance': spaces.Box(low=0, high=np.sqrt(env_params['rel_dist'] ** 2 + env_params['rel_dist'] ** 2), shape=(1,),
                                   dtype=np.float64),
            'rel_bearing': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
            'flow_field': spaces.Box(
                low=np.array([[env_params['alt_min'], 0, min_vel]] * num_levels, dtype=np.float64),
                high=np.array([[env_params['alt_max'], np.pi, max_vel]] * num_levels, dtype=np.float64),
                dtype=np.float64)
        })


    def seed(self, seed=None):
        """
        Need to assign numpy random number generator in cases multiple envelopes are selected.
        """
        if seed!=None:
            self.np_rng = np.random.default_rng(seed)
        else:
            self.np_rng = np.random.default_rng(np.random.randint(0, 2**32))

    def count_greater_than_zero(self, arr):
        return np.sum(np.array(arr) > 0)

    def reset(self, seed=None, options=None):

        #Randomize new coordinate and forecast subset
        #self.forecast_score = 0
        #while self.forecast_score< 0.25:
        #self.forecast_subset.randomize_coord()
        #self.forecast_subset.subset_forecast()

        self.forecast_scores = [5, 5, 5, 5]  # dummy score to trigger randomizing
        self.forecast_score = -1  # dummy score to trigger randomizing

        #while self.count_greater_than_zero(self.forecast_scores) != 0:

        # For not including bad forecasts (score of 0):
        #'''
        while self.forecast_score < env_params['forecast_score_threshold']:
        #while not (self.forecast_score > .45 and self.forecast_score < .55):

            self.forecast_subset_era5.randomize_coord()
            self.forecast_subset_era5.subset_forecast(days=self.days)

            # Then assign coord to synth winds
            self.forecast_subset_synth.assign_coord(lat=self.forecast_subset_era5.lat_central,
                                                    lon=self.forecast_subset_era5.lon_central,
                                                    timestamp=self.forecast_subset_era5.start_time)
            self.forecast_subset_synth.subset_forecast(days=self.days)
        

            self.forecast_scores, self.forecast_score = self.ForecastClassifier.determine_OW_Rate(self.forecast_subset_era5)
        #'''

        #For including bad forecasts (score of 0):
        '''
        self.forecast_subset.randomize_coord()
        self.forecast_subset.subset_forecast(days=self.days)
        self.forecast_scores, self.forecast_score = self.ForecastClassifier.determine_OW_Rate(self.forecast_subset)
        '''

        #print(self.forecast_scores)
        #if score < 0.25:
        #   print(colored("WARNING: Bad forecast score of " + str(score) + ". Re-randomizing" , "yellow"))
        #if self.count_greater_than_zero(self.forecast_scores) != 0:
        #    print(colored("WARNING: Bad forecast super score of " + str(self.count_greater_than_zero(self.forecast_scores)) + ". Re-randomizing" , "yellow"))

        #Reset custom metrics
        self.within_target = False
        self.twr = 0 # time within radius
        self.twr_inner = 0  # time within radius
        self.twr_outer = 0  # time within radius

        #Reset Balloon State to forecast subset central point.
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

        self.Balloon.update(lat = self.forecast_subset_era5.lat_central, lon = self.forecast_subset_era5.lon_central, x=0, y=0, distance = 0)
        self.calculate_relative_wind_column() #?????????????????

        if self.render_mode == "human":
            self.renderer.reset(self.goal, self.Balloon, self.SimulatorState)
            self.Forecast_visualizer.generate_flow_array(self.forecast_subset_era5.start_time)  # change this for time?

        return self._get_obs(), self._get_info()

    """
    @profile
    def getCoord_ERA5(self,coord,dt):
        return self.gfs.getNewCoord(coord, dt)

    @profile
    def getCoord_XR(self,coord,dt):
        #vertical_column = self.forecast.ds.sel(latitude=self.Balloon.lat, longitude=self.Balloon.lat,
        #                                       time=self.SimulatorState.timestamp, method='nearest')

        vertical_column = self.forecast_subset.ds.isel(latitude=5, longitude=5,
                                               time=5)

        alt_column = vertical_column['z'].values[::-1] / constants.GRAVITY
        u_column = vertical_column['u'].values[::-1]
        v_column = vertical_column['v'].values[::-1]

        u = np.interp(self.Balloon.altitude, alt_column, u_column)
        v = np.interp(self.Balloon.altitude, alt_column, v_column)

        print(u,v)

    @profile
    def getCoord_forecast(self):
        return self.forecast_subset.getNewCoord(self.Balloon, self.SimulatorState, self.dt)
    """

    def move_agent(self, action):
        """
        Altitude is currently capped to not be able to go out of bounds.

        """
        #self.Balloon.lat,self.Balloon.lon,self.Balloon.x_vel,self.Balloon.y_vel, _, _, _,_, _, _ = self.getCoord_ERA5(coord,self.dt)
        #self.getCoord_XR(coord, self.dt)

        # Look up flow at current 3D position before altitude change. Update Position and Flow State
        #LOOK UP MOVEMENT IN SYNTH instead of ERA5
        self.Balloon.lat,self.Balloon.lon,self.Balloon.x_vel,self.Balloon.y_vel, self.Balloon.x, self.Balloon.y = self.forecast_subset_synth.getNewCoord(self.Balloon, self.SimulatorState, self.dt)

        self.Balloon.distance = np.sqrt((self.Balloon.x - self.goal["x"]) ** 2 + (self.Balloon.y - self.goal["y"]) ** 2)
        self.Balloon.rel_bearing = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                                 self.goal["x"], self.goal["y"],
                                                                 self.Balloon.x_vel, self.Balloon.y_vel)

        self.Balloon.rel_wind_column = self.calculate_relative_wind_column()

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

        return 0

    def reward_google(self):
        distance_to_target = self.Balloon.distance
        c_cliff = 0.4
        tau = 100

        if self.Balloon.altitude >= env_params['alt_min'] and self.Balloon.altitude <= env_params['alt_max']:

            if distance_to_target <= self.radius:
                reward = 1
                self.twr += 1
                self.within_target = True
            else:
                #reward = np.exp(-0.01 * (distance_to_target - self.radius))
                reward = c_cliff*2*np.exp((-1*(distance_to_target-self.radius)/tau))
                self.within_target = False

            #Add more regions to track,  Not doing anything with them yet,  just for metric analysis
            if distance_to_target <= self.radius_inner:
                self.twr_inner += 1

            if distance_to_target <= self.radius_outer:
                self.twr_outer += 1

        else:
            reward = 0

        return reward

    def reward_piecewise(self):
        '''
        Extra reward possible for station keeping within inner necessary, otherwise following google's structure
        :return:
        '''
        distance_to_target = self.Balloon.distance
        c_cliff = 0.4
        tau = 100

        if self.Balloon.altitude >= env_params['alt_min'] and self.Balloon.altitude <= env_params['alt_max']:

            if distance_to_target <= self.radius_inner:
                reward = 2
                self.twr += 1
                self.within_target = True
            elif distance_to_target <= self.radius and distance_to_target > self.radius_inner:
                reward = 1
                self.twr += 1
                self.within_target = True
            else:
                # reward = np.exp(-0.01 * (distance_to_target - self.radius))
                reward = c_cliff * 2 * np.exp((-1 * (distance_to_target - self.radius) / tau))
                self.within_target = False

            # Add more regions to track,  Not doing anything with them yet,  just for metric analysis
            if distance_to_target <= self.radius_inner:
                self.twr_inner += 1

            if distance_to_target <= self.radius_outer:
                self.twr_outer += 1

        else:
            reward = 0

        return reward

    def reward_euclidian(self):
        '''
        Linear Euclidian reward within target region, google cliff function for outside of radius

        '''

        distance_to_target = self.Balloon.distance
        c_cliff = 0.4
        tau = 100
        self.within_target = False #by default

        if self.Balloon.altitude >= env_params['alt_min'] and self.Balloon.altitude <= env_params['alt_max']:

            if distance_to_target <= self.radius:
                #Normalize distance within radius,  for a maximum score of 2.
                reward = convert_range(distance_to_target,0,self.radius, 2, 1)
                self.twr += 1
                self.within_target = True

            else:
                # reward = np.exp(-0.01 * (distance_to_target - self.radius))
                reward = c_cliff * 2 * np.exp((-1 * (distance_to_target - self.radius) / tau))
                self.within_target = False

            # Add more regions to track,  Not doing anything with them yet,  just for metric analysis
            if distance_to_target <= self.radius_inner:
                self.twr_inner += 1

            if distance_to_target <= self.radius_outer:
                self.twr_outer += 1

        #no reward for going outside of altitude control bounds
        else:
            reward = 0

        #print(self.within_target, "Reward", reward)

        return reward

    def step(self, action):
        reward = self.move_agent(action)
        reward += self.reward_piecewise()

        observation = self._get_obs()
        info = self._get_info()

        done = self.SimulatorState.step(self.Balloon)

        #if done:
        #    print(observation, reward, done, False, info)

        return observation, reward, done, False, info

    def calculate_relative_angle(self, x, y, goal_x, goal_y, heading_x, heading_y):
        """
        Calculates the relative angle of motion of the blimp in relation to the goal based off of true bearing FROM POSITION TO GOAL
        and the true heading.  The relative angle of motion is then between 0 and 180 degrees, where...

        If the balloon moves directly to the goal at one altitude, the value would be 0 degrees.  Alternatively, if the balloon moved directly
        away from the goal at another altitude level, that would be 180 degrees.

        :param x: current balloon x position
        :param y: current balloon y position
        :param goal_x:
        :param goal_y:
        :param heading_x: current balloon x velocity
        :param heading_y: current balloon y velocity
        :return: relative bearing of balloon motion in relation to goal
        """

        # Calculate the current heading based on the heading vector
        heading = np.arctan2(heading_y, heading_x)

        # Calculate the true (inverted) bearing FROM POSITION TO GOAL
        true_bearing = np.arctan2(goal_y - y, goal_x - x)

        # Find the absolute difference between the two angles
        rel_bearing = np.abs(heading - true_bearing)

        # map from [-pi, pi] to [0,pi]
        rel_bearing = abs((rel_bearing + np.pi) % (2 * np.pi) - np.pi)

        return rel_bearing

    def calculate_relative_wind_column(self):
        """
        Builds off of the same calculation as calculate_relative_angle() to calculate the relative
        "flow map" vertical slice from the balloons current position between 0 and 180 degrees.

        (z, magnitude, bearing)  add uncertatinty later)

        :return: flowfield with relative bearins and magnitude
        """

        #First need to get altitude coordinate from forecast
        #vertical_column = self.forecast.ds.sel(latitude=self.Balloon.lat, longitude=self.Balloon.lat, time= self.SimulatorState.timestamp, method='nearest')

        #alt_column = vertical_column['z'].values[::-1] / constants.GRAVITY
        #u_column = vertical_column['u'].values[::-1]
        #v_column = vertical_column['v'].values[::-1]

        alt_column,u_column,v_column = self.forecast_subset_era5.np_lookup(self.Balloon.lat, self.Balloon.lon, self.SimulatorState.timestamp)

        alt_column = alt_column[::-1]
        u_column = u_column[::-1]
        v_column = v_column[::-1]

        flow_field_rel_angle = []
        flow_field_magnitude = []

        for i in range (0,len(alt_column)):
            rel_angle = self.calculate_relative_angle(self.Balloon.x, self.Balloon.y,
                                                      self.goal["x"], self.goal["y"], u_column[i], v_column[i])
            magnitude = math.sqrt(u_column[i]**2 + v_column[i]**2)

            flow_field_rel_angle.append(rel_angle)
            flow_field_magnitude.append(magnitude)

        #Relative Flow Field Map
        return np.stack((alt_column, flow_field_rel_angle, flow_field_magnitude), axis=-1)

    def _get_obs(self):
        observation = {
            'altitude': np.array([self.Balloon.altitude]),
            'distance': np.array([self.Balloon.distance]),
            'rel_bearing': np.array([self.Balloon.rel_bearing]),
            'flow_field': self.Balloon.rel_wind_column
        }

        return observation

    def _get_info(self):
        return {
            "distance": self.Balloon.distance,
            "within_target": self.within_target,
            "twr": self.twr,
            "twr_inner": self.twr_inner,
            "twr_outer": self.twr_outer,
            "forecast_score": self.forecast_score,
            "forecast_scores": self.forecast_scores,
            "render_mode": self.render_mode
        }

    def render(self, mode='human'):
        self.renderer.render(mode='human')

    def close(self):
        pass

# For keyboard control
def on_press(key):
    global last_action
    try:
        if key == keyboard.KeyCode.from_char("w"):
            last_action = command.UP
        elif key == keyboard.KeyCode.from_char("d"):
            last_action = command.STAY
        elif key == keyboard.KeyCode.from_char("x"):
            last_action = command.DOWN
        else:
            last_action = command.STAY  # Default action when no arrow key is pressed
    except AttributeError:
        pass

# Global variable to store the last action pressed
last_action = command.STAY  # Default action
listener = keyboard.Listener(on_press=on_press)
listener.start()

def main():
    #np.set_printoptions(threshold=sys.maxsize)
    #filename = "July-2024-SEA.nc"
    #filename = "SYNTH-Jan-2023-SEA.nc"

    FORECAST_SYNTH = Forecast(env_params['synth_netcdf'], forecast_type="SYNTH")
    # Get month associated with Synth
    month = pd.to_datetime(FORECAST_SYNTH.TIME_MIN).month
    # Then process ERA5 to span the same timespan as a monthly Synthwinds File
    FORECAST_ERA5 = Forecast(env_params['era_netcdf'], forecast_type="ERA5", month=month)

    env = FlowFieldEnv3d_DUAL(FORECAST_ERA5=FORECAST_ERA5, FORECAST_SYNTH=FORECAST_SYNTH, render_mode=env_params['render_mode'])

    while True:
        start_time = time.time()

        obs, info = env.reset()
        total_reward = 0
        for step in range( env_params["episode_length"]+10):

            #print()
            #print(obs)

            # Use this for keyboard input
            obs, reward, done, truncated, info = env.step(last_action)
            total_reward += reward


            if env_params['render_mode'] == "human":
                env.render()

            if done:
                break


            #sys.exit()
            #time.sleep(2)
        # print(obs)

        print("Total reward:", total_reward, info, env_params['seed'])
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time)

if __name__ == '__main__':
    main()
