import matplotlib.pyplot as plt
import numpy as np
from env.config.env_config import env_params
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

class MatplotlibRendererTriple():
    def __init__(self, Forecast_visualizer_ERA5, Forecast_visualizer_SYNTH,  render_mode,
                 radius, coordinate_system = "geographic"):

        self.coordinate_system = coordinate_system

        self.Forecast_visualizer = Forecast_visualizer_ERA5

        self.Forecast_visualizer_synth = Forecast_visualizer_SYNTH

        self.render_count = env_params['render_count']
        self.render_skip = env_params['render_skip']
        self.render_mode = render_mode

        self.render_timestamp = self.Forecast_visualizer.forecast_subset.start_time

        self.render_timestamp_synth = self.Forecast_visualizer_synth.forecast_subset.start_time

        self.dt = env_params['dt']
        self.episode_length = env_params['episode_length']


        self.goal = {"x": 0, "y": 0} #relative

        self.radius = radius  # m
        self.radius_inner = radius * .5  # m
        self.radius_outer = radius * 1.5  # m

        #try:
        if self.coordinate_system == "geographic":
            #Zone 12 for Albuquerque. Will need to change this for other areas
            #self.p = Proj(proj='utm', zone=12, ellps='WGS84', preserve_units=False)

            #Also Central Coord for now?
            #self.start_coord = config_earth.simulation['start_coord']
            #x, y = self.p(longitude=self.start_coord["lon"], latitude=self.start_coord["lat"])

            self.init_plot_geographic()

        if self.coordinate_system == "cartesian":

            self.init_plot()
        #except:
            #print(colored("Not a Valid Coordinate System. Can either be geographic or cartesian","red"))

    def init_plot_geographic(self):
        self.fig = plt.figure(figsize=(18, 10))



        #self.ax.set_title("3D Trajectory")


        #self.gs = self.fig.add_gridspec(nrows=2, ncols=3, height_ratios=[1, 4])

        self.gs = self.fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1, 4], width_ratios=[.9, .9, .9, 0.05])

        # Top Altitude Plot
        self.ax3 = self.fig.add_subplot(self.gs[0, :])

        # 3D plots
        self.ax = self.fig.add_subplot(self.gs[1, 0], projection='3d')
        self.ax2 = self.fig.add_subplot(self.gs[1, 1], projection='custom3dquiver')
        self.ax4 = self.fig.add_subplot(self.gs[1, 2], projection='custom3dquiver')


        # Adjust the space between rows (hspace = 0 for no space)
        self.gs.update(hspace=0)

        box1 = self.ax.get_position()  # Get the original position of the top plot
        box2 = self.ax3.get_position()  # Get the original position of the bottom plot

        # Adjust the position of the bottom plot to overlap the top one slightly
        self.ax2.set_position([box2.x0, box2.y0 - 0.05, box2.width, box2.height])

        self.ax.set_xlabel('Relative X (km)')
        self.ax.set_ylabel('Relative Y (km)')
        self.ax.set_zlabel('Altitude (km)')


        self.ax.set_xlim(-env_params['rel_dist'], env_params['rel_dist'])
        self.ax.set_ylim(-env_params['rel_dist'], env_params['rel_dist'])
        self.ax.set_zlim(env_params['alt_min'], env_params['alt_max'])
        self.ax.set_title("3D Trajectory")

        xticks = self.ax.get_xticks()  # Get current tick positions
        self.ax.set_xticks(xticks)  # Fix the tick locations
        self.ax.set_xticklabels([f'{int(tick / 1000)}' for tick in xticks])  # Divide by 100 and set
        # For Y axis
        yticks = self.ax.get_yticks()
        self.ax.set_yticks(yticks)  # Fix the tick locations
        self.ax.set_yticklabels([f'{int(tick / 1000)}' for tick in yticks])
        zticks = self.ax.get_zticks()
        self.ax.set_zticks(zticks)  # Fix the tick locations
        self.ax.set_zticklabels([f'{int(tick / 1000)}' for tick in zticks])




        self.path_plot, = self.ax.plot([], [], [], color='black')
        self.scatter = self.ax.scatter([], [], [], color='black')
        self.ground_track, = self.ax.plot([], [], [], color='red')
        self.scatter_goal = self.ax.scatter([], [], [], color='green')
        self.canvas = self.fig.canvas

        self.Forecast_visualizer.visualize_3d_planar_flow(self.ax2, skip=self.render_skip)

        self.Forecast_visualizer_synth.visualize_3d_planar_flow(self.ax4, skip=self.render_skip)

        self.current_state_line, = self.ax.plot([], [], [], 'r--')

        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius, color='g-')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_inner, color='g--')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_outer, color='g--')

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

    def reset(self, goal, Balloon, SimulatorState):
        if hasattr(self, 'fig'):
            plt.close('all')
            delattr(self, 'fig')
            delattr(self, 'ax')
            delattr(self, 'ax2')
            delattr(self, 'ax3')
            delattr(self, 'ax4')
            delattr(self, 'goal')
            delattr(self, 'scatter')
            delattr(self, 'canvas')

        self.Balloon = Balloon
        self.SimulatorState = SimulatorState
        self.goal = goal

        self.render_step = 1
        self.hour_count = 0

        self.render_timestamp = self.Forecast_visualizer.forecast_subset.start_time
        self.render_timestamp_synth = self.Forecast_visualizer_synth.forecast_subset.start_time


    def plot_circle(self, ax, center_x,center_y, radius, plane='xy', color ='g--'):
        #UPDATE: This is a new function because the radius wasn't plotting properly for smaller radii
        # Create the angle array
        theta = np.linspace(0, 2 * np.pi, 100)

        # Generate the circle points in 2D
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)

        if plane == 'xy':
            x = center_x + circle_x
            y = center_y + circle_y
            z = np.full_like(x, env_params['alt_min'])

        ax.plot(x, y, z, color)

    def render(self, mode='human'):

        if not hasattr(self, 'fig'):
            if self.coordinate_system == "geographic":
                self.init_plot_geographic()

            if self.coordinate_system == "cartesian":
                self.init_plot()

        if self.render_step == self.render_count:

            path = np.array(self.SimulatorState.trajectory)

            self.path_plot.set_data(np.array(path)[:, :2].T)
            self.path_plot.set_3d_properties(np.array(path)[:, 2])

            self.ground_track.set_data(np.array(path)[:, :2].T)
            self.ground_track.set_3d_properties(np.full(len(path), env_params['alt_min']))

            self.scatter._offsets3d = (
            np.array([self.Balloon.x]), np.array([self.Balloon.y]), np.array([self.Balloon.altitude]))
            self.scatter_goal._offsets3d = (np.array([self.goal["x"]]), np.array([self.goal["y"]]), np.array([env_params['alt_min']]))

            self.current_state_line.set_data([self.Balloon.x, self.Balloon.x], [self.Balloon.y, self.Balloon.y])
            self.current_state_line.set_3d_properties([env_params['alt_min'], self.Balloon.altitude])

            self.altitude_line.set_data(range(len(path)), path[:, 2]/1000.)

            self.canvas.draw()
            # self.canvas.flush_events()

            self.ax3.set_title("Timestamp: " + str(self.SimulatorState.timestamp) + "\nTime Elapsed: " + str((self.SimulatorState.timestamp - self.render_timestamp)))




            duration_in_s = (self.SimulatorState.timestamp - self.render_timestamp).total_seconds()
            self.hours = int(divmod(duration_in_s, 3600)[0])


            if self.hours > self.hour_count:
                self.ax2.clear()
                self.ax2.remove()

                self.ax2 = self.fig.add_subplot(self.gs[1, 1], projection='custom3dquiver')
                self.Forecast_visualizer.generate_flow_array(timestamp=self.SimulatorState.timestamp)
                self.Forecast_visualizer.visualize_3d_planar_flow(self.ax2, skip=self.render_skip)

                self.ax4.clear()
                self.ax4.remove()

                self.ax4 = self.fig.add_subplot(self.gs[1, 2], projection='custom3dquiver')
                self.Forecast_visualizer_synth.generate_flow_array(timestamp=self.SimulatorState.timestamp)
                self.Forecast_visualizer_synth.visualize_3d_planar_flow(self.ax4, skip=self.render_skip)



                self.hour_count += 1

            if mode == 'human':
                plt.pause(0.001)

                # For saving plots in the paper
                #plt.show()
                #pass

            self.render_step = 1

        else:
            self.render_step += 1