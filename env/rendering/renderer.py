import matplotlib.pyplot as plt
import numpy as np
from env.config.env_config import env_params
from env.rendering.Trajectory3DPlotter import Trajectory3DPlotter
from termcolor import colored

class MatplotlibRenderer():
    def __init__(self, Forecast_visualizer, render_mode,
                 radius, coordinate_system = "geographic"):

        self.coordinate_system = coordinate_system

        self.Forecast_visualizer = Forecast_visualizer

        self.render_count = env_params['render_count']
        self.quiver_skip = env_params['quiver_skip']
        self.render_mode = render_mode

        self.render_timestamp = self.Forecast_visualizer.forecast_subset.start_time

        self.dt = env_params['dt']
        self.episode_length = env_params['episode_length']

        self.goal = {"x": 0, "y": 0} #relative

        self.radius = radius  # m
        self.radius_inner = radius * .5  # m
        self.radius_outer = radius * 1.5  # m

        try:
            if self.coordinate_system == "geographic":
                self.init_plot_geographic()

            if self.coordinate_system == "cartesian":
                #this does not exist right now
                self.init_plot()
        except:
            print(colored("Not a Valid Coordinate System. Can either be geographic or cartesian","red"))

    def init_plot_geographic(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.gs = self.fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 4])
        self.ax3 = self.fig.add_subplot(self.gs[0, :])
        self.ax = self.fig.add_subplot(self.gs[1, 0], projection='3d')
        self.ax2 = self.fig.add_subplot(self.gs[1, 1], projection='custom3dquiver')

        # Initialize the 3D trajectory plotter here
        self.trajectory_plotter = Trajectory3DPlotter(self.ax, self.radius, self.goal, self.dt, self.episode_length)

        self.Forecast_visualizer.visualize_3d_planar_flow(self.ax2, quiver_skip=self.quiver_skip)

        # Altitude Profile Plot Setup
        self.altitude_line, = self.ax3.plot([], [], 'b-')
        self.ax3.set_xlabel('Number of Steps (dt=' + str(self.dt) + ')')
        self.ax3.set_ylabel('Altitude (m)')
        self.ax3.set_xlim(0, self.episode_length)
        self.ax3.set_ylim(env_params['alt_min'],env_params['alt_max']+100)

    def reset(self, goal, Balloon, SimulatorState):
        if hasattr(self, 'fig'):
            plt.close('all')
            delattr(self, 'fig')
            delattr(self, 'ax')
            #delattr(self, 'ax2')
            delattr(self, 'ax3')
            delattr(self, 'goal')
            #delattr(self, 'scatter')
            #delattr(self, 'canvas')

        self.Balloon = Balloon
        self.SimulatorState = SimulatorState
        self.goal = goal

        self.render_step = 1
        self.hour_count = 0

        self.render_timestamp =  self.SimulatorState.timestamp    #self.Forecast_visualizer.forecast_subset.start_time

    def is_timestamp_in_interval(self, timestamp, interval):
        """
        Maybe move this to utils

        Checks if the provided timestamp is in intervals of 3, 6, or 12 hours..
        """
        # Convert the timestamp string to a datetime object
        #dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

        # Get the hour from the datetime object
        hour = timestamp.hour
        minute = timestamp.minute

        # Check if the minutes are exactly 00
        if minute != 0:
            return False

        # Check if the hour is divisible by the interval
        if interval == 3:
            return hour % 3 == 0
        elif interval == 6:
            return hour % 6 == 0
        elif interval == 12:
            return hour % 12 == 0
        else:
            raise ValueError("Interval must be 3, 6, or 12 hours.")

    def render(self, mode='human'):

        if not hasattr(self, 'fig'):
            if self.coordinate_system == "geographic":
                self.init_plot_geographic()

            if self.coordinate_system == "cartesian":
                self.init_plot()

        if self.render_step == self.render_count:
            # Update path data for drawing 3D trajectory and altitude profile
            path = np.array(self.SimulatorState.trajectory)

            # Update the 3D trajectory plot
            self.trajectory_plotter.update(self.Balloon, path, self.goal)
            self.trajectory_plotter.update_altitude(path)
            self.trajectory_plotter.draw()

            # Update Altitude Profile
            self.altitude_line.set_data(range(len(path)), path[:, 2])
            self.ax3.set_title("Timestamp: " + str(self.SimulatorState.timestamp) + "\nTime Elapsed: " + str((self.SimulatorState.timestamp - self.render_timestamp)))

            # Check if timestamp is in a 3 hour interval (assume timewarping) If so update flow visualization
            if self.is_timestamp_in_interval(self.SimulatorState.timestamp, 3):
                # Handle visualizing the 3D planar flow
                self.ax2.clear()
                self.Forecast_visualizer.generate_flow_array(timestamp=self.SimulatorState.timestamp)
                self.Forecast_visualizer.visualize_3d_planar_flow(self.ax2, quiver_skip=self.quiver_skip)


            if mode == 'human':
                plt.pause(0.001)

            self.render_step = 1

        else:
            self.render_step += 1