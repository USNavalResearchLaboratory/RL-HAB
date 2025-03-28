import matplotlib.pyplot as plt
import numpy as np

#from Plotting.era5_forecast_visualizations import skip_array
#dfdg
from env.config.env_config import env_params
from env.rendering.Trajectory3DPlotter import Trajectory3DPlotter
import matplotlib as mpl
from termcolor import colored
import os

class MatplotlibRendererTriple():
    """
    Handles the visualization of ERA5 and Synthetic forecasts using 3D quiver plots and trajectory visualization.

    Attributes:
        coordinate_system (str): Coordinate system used for visualization ('geographic' or 'cartesian').
        Forecast_visualizer (object): Visualizer for ERA5 forecasts.
        Forecast_visualizer_synth (object): Visualizer for synthetic forecasts.
        render_count (int): Frequency of rendering updates.
        quiver_skip (int): Frequency of skipping quivers in plots for performance.
        render_mode (str): Rendering mode ('human', etc.).
        render_timestamp (numpy.datetime64): Timestamp for ERA5 rendering.
        render_timestamp_synth (numpy.datetime64): Timestamp for synthetic rendering.
        dt (float): Time step of the simulation.
        episode_length (int): Total steps in an episode.
        goal (dict): Goal position (x, y).
        radius (float): Station-keeping radius.
        radius_inner (float): Inner radius for goal visualization.
        radius_outer (float): Outer radius for goal visualization.
        trajectory_plotter (Trajectory3DPlotter): Handles trajectory visualization.
    """
    def __init__(self, Forecast_visualizer_ERA5, Forecast_visualizer_SYNTH,  render_mode,
                 radius, coordinate_system = "geographic"):
        """
        Initialize the renderer.

        Args:
            Forecast_visualizer_ERA5 (object): Visualizer for ERA5 forecast.
            Forecast_visualizer_SYNTH (object): Visualizer for synthetic forecast.
            render_mode (str): Rendering mode ('human', etc.).
            radius (float): Station-keeping radius in meters.
            coordinate_system (str, optional): Coordinate system ('geographic' or 'cartesian').
        """

        self.coordinate_system = coordinate_system

        self.Forecast_visualizer = Forecast_visualizer_ERA5
        self.Forecast_visualizer_synth = Forecast_visualizer_SYNTH
        

        self.render_count = env_params['render_count']
        self.quiver_skip = env_params['quiver_skip']
        self.render_mode = render_mode

        self.render_timestamp = self.Forecast_visualizer.forecast_subset.start_time
        self.render_timestamp_synth = self.Forecast_visualizer_synth.forecast_subset.start_time

        self.dt = env_params['dt']
        self.episode_length = env_params['episode_length']

        self.goal = {"x": 0, "y": 0} #relative

        self.radius = radius  # m
        self.radius_inner = radius * .5  # m
        self.radius_outer = radius * 1.5  # m

        self.save_figure = env_params['save_figure']

        if self.save_figure:
            self.frame_save_dir = env_params['save_dir']
            os.makedirs(self.frame_save_dir, exist_ok=True)
            self.frame_count = 0


        #try:
        if self.coordinate_system == "geographic":
            self.init_plot_geographic()

        if self.coordinate_system == "cartesian":
            # this does not exist right now
            self.init_plot()
        #except:
            #print(colored("Not a Valid Coordinate System. Can either be geographic or cartesian", "red"))
            

    def init_plot_geographic(self):
        """
        Initialize the plot layout for geographic visualization.
        """
        self.fig = plt.figure(figsize=(18, 10))
        self.gs = self.fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1, 4], width_ratios=[.9, .9, .9, 0.05])

        # Top Altitude Plot
        self.ax3 = self.fig.add_subplot(self.gs[0, :])

        # 3D plots
        self.ax = self.fig.add_subplot(self.gs[1, 0], projection='3d')
        self.ax2 = self.fig.add_subplot(self.gs[1, 1], projection='custom3dquiver') # ERA5
        self.ax4 = self.fig.add_subplot(self.gs[1, 2], projection='custom3dquiver') # Synth

        # Adjust the space between rows (hspace = 0 for no space)
        self.gs.update(hspace=0)

        #box1 = self.ax.get_position()  # Get the original position of the top plot
        box2 = self.ax3.get_position()  # Get the original position of the bottom plot

        # Adjust the position of the bottom plot to overlap the top one slightly
        self.ax2.set_position([box2.x0, box2.y0 - 0.05, box2.width, box2.height])

        #These 3 lines make it so the first ERA5 forecast visualization is not glitchy during rendering.  Not sure why it is in the first place.
        self.ax2.clear()
        self.ax2.remove()
        self.ax2 = self.fig.add_subplot(self.gs[1, 1], projection='custom3dquiver')

        # Initialize the 3D trajectory plotter here
        self.trajectory_plotter = Trajectory3DPlotter(self.ax, self.radius, self.goal, self.dt, self.episode_length)

        # Initialize 2 Forecast visualizers, 1 for ERA5, one for Synth
        self.Forecast_visualizer.visualize_3d_planar_flow(self.ax2, quiver_skip=self.quiver_skip)
        self.Forecast_visualizer_synth.visualize_3d_planar_flow(self.ax4, quiver_skip=self.quiver_skip)

        # Altitude Profile Plot Setup
        self.altitude_line, = self.ax3.plot([], [], 'b-')
        self.ax3.set_xlabel('Number of Steps (dt=' + str(self.dt) + 's)')
        self.ax3.set_ylabel('Altitude (km)')
        self.ax3.set_xlim(0, self.episode_length)
        self.ax3.set_ylim(env_params['alt_min']/1000.,env_params['alt_max']/1000.)

        # Manual COlor Bar stuff:
        plt.title('')

        # Manual colorbar creation from -180° to 180°
        norm = mpl.colors.Normalize(vmin=-180, vmax=180)  # Normalize the range from -180 to 180
        sm = mpl.cm.ScalarMappable(cmap='hsv', norm=norm)  # ScalarMappable ties the colormap and normalization together
        sm.set_array([])  # Required for ScalarMappable (no actual data, so empty array)

        # Create a new axis for the colorbar with manual size
        # [left, bottom, width, height] where values are fractions of figure size
        cbar_ax = self.fig.add_axes([0.89, 0.25, 0.01, 0.35])  # Half-height colorbar

        # Create colorbar in this new axis
        cbar = self.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Wind Direction (degrees)')

        # Remove any title from the colorbar
        # print("title",cbar_ax.title())
        #cbar_ax.title.set_text('First Plot')
        # sdfsdf
        # self.cbar_ax.set_title('')  # Ensures no title appears

        # Set custom ticks and labels (with degree symbols)
        cbar.set_ticks([-180, -90, 0, 90, 180])
        cbar.set_ticklabels([r'$-180^\circ$', r'$-90^\circ$', r'$0^\circ$', r'$90^\circ$', r'$180^\circ$'])

        #self.reset(self.Balloon, self.goal, self.SimulatorState)

    def reset_forecast_visualization(self):
        """
        Reset the forecast visualization for ERA5 and Synthetic forecasts.
        """
        self.ax2.clear()
        self.ax2.remove()

        self.ax2 = self.fig.add_subplot(self.gs[1, 1], projection='custom3dquiver')
        self.Forecast_visualizer.generate_flow_array(timestamp=self.SimulatorState.timestamp)
        self.Forecast_visualizer.visualize_3d_planar_flow(self.ax2, quiver_skip=self.quiver_skip,
                                                          altitude_quiver_skip=1)

        self.ax4.clear()
        self.ax4.remove()

        self.ax4 = self.fig.add_subplot(self.gs[1, 2], projection='custom3dquiver')
        self.Forecast_visualizer_synth.generate_flow_array(timestamp=self.SimulatorState.timestamp)
        self.Forecast_visualizer_synth.visualize_3d_planar_flow(self.ax4, quiver_skip=self.quiver_skip)

    def reset(self, goal, Balloon, SimulatorState):
        """
        Reset the renderer.

        Args:
            goal (dict): Goal position (x, y).
            Balloon (object): Balloon state object.
            SimulatorState (object): Simulator state object.
        """
        #close figures if already open, otherwise make them
        if hasattr(self, 'fig'):
            plt.close('all')
            delattr(self, 'fig')
            delattr(self, 'ax')
            #delattr(self, 'ax2')
            delattr(self, 'ax3')
            #delattr(self, 'ax4')
            delattr(self, 'goal')
            #delattr(self, 'scatter')
            #delattr(self, 'canvas')


        self.Balloon = Balloon
        self.SimulatorState = SimulatorState
        self.goal = goal

        self.render_step = 1 #for initial reset
        self.hour_count = 0

        #self.render_timestamp = self.Forecast_visualizer.forecast_subset.start_time
        #self.render_timestamp_synth = self.Forecast_visualizer_synth.forecast_subset.start_time

        self.render_timestamp = self.SimulatorState.timestamp    #self.Forecast_visualizer.forecast_subset.start_time

        if not hasattr(self, 'fig'):
            if self.coordinate_system == "geographic":
                self.init_plot_geographic()

            if self.coordinate_system == "cartesian":
                self.init_plot()

        self.reset_forecast_visualization()

    def is_timestamp_in_interval(self, timestamp, interval):
        """
        Check if a timestamp falls within a specified interval.

        Args:
            timestamp (datetime.datetime): Timestamp to check.
            interval (int): Interval in hours.

        Returns:
            bool: True if timestamp falls within the interval, False otherwise.
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
        if interval == 1:
            return hour % 1 == 0
        if interval == 3:
            return hour % 3 == 0
        elif interval == 6:
            return hour % 6 == 0
        elif interval == 12:
            return hour % 12 == 0
        else:
            raise ValueError("Interval must be 3, 6, or 12 hours.")


    def render(self, mode='human'):
        """
        Render the simulation environment.

        Args:
            mode (str, optional): Rendering mode ('human', etc.).
        """

        if self.render_step == self.render_count:

            path = np.array(self.SimulatorState.trajectory)

            # Update the 3D trajectory plot
            self.trajectory_plotter.update(self.Balloon, path, self.goal)
            self.trajectory_plotter.update_altitude(path)
            self.trajectory_plotter.draw()

            # Update Altitude Profile
            self.altitude_line.set_data(range(len(path)), path[:, 2]/1000.) #Doing this in km
            self.ax3.set_title("Timestamp: " + str(self.SimulatorState.timestamp) + "\nTime Elapsed: " + str(
                (self.SimulatorState.timestamp - self.render_timestamp)))

            if env_params['timewarp'] == None:
                forecast_visualizer_update = 12
            else:
                forecast_visualizer_update = env_params['timewarp']

            if self.is_timestamp_in_interval(self.SimulatorState.timestamp, forecast_visualizer_update):
                self.reset_forecast_visualization()

            if mode == 'human':
                plt.pause(0.001)
                print(self.SimulatorState.timestamp, self.Balloon)

            # Save frame as image
            if self.save_figure:
                frame_filename = os.path.join(self.frame_save_dir, f"frame_{self.frame_count:05d}.png")
                self.fig.savefig(frame_filename)
                self.frame_count += 1

            self.render_step = 1

        else:
            self.render_step += 1