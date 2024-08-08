import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
from pyproj import Proj
from utils import CoordinateTransformations as transform

from era5 import config_earth
from env3d.config.env_config import env_params


class MatplotlibRenderer():
    def __init__(self, x_dim, y_dim, z_dim, FlowField3d, render_count, render_skip, render_mode,
                 radius, dt, episode_length, coordinate_system = "cartesian"):

        self.coordinate_system = coordinate_system
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.FlowField3D = FlowField3d
        self.render_count = render_count
        self.render_skip = render_skip
        self.render_mode = render_mode

        self.dt = dt
        self.episode_length = episode_length

        #try:
        if self.coordinate_system == "geographic":
            #Zone 12 for Albuquerque. Will need to change this for other areas
            self.p = Proj(proj='utm', zone=12, ellps='WGS84', preserve_units=False)

            #Also Central Coord for now?
            self.start_coord = config_earth.simulation['start_coord']
            x, y = self.p(longitude=self.start_coord["lon"], latitude=self.start_coord["lat"])

            self.goal = {"x": x, "y": y}


            self.radius = radius * 1000  # m
            self.radius_inner = radius * .5 * 1000  # m
            self.radius_outer = radius * 1.5 * 1000  # m

            self.init_plot_geographic()

        if self.coordinate_system == "cartesian":
            self.goal = {"x": 0, "y": 0}  # Dummy numbers to start

            self.radius = radius   # m
            self.radius_inner = radius * .5   # m
            self.radius_outer = radius * 1.5  # m

            self.init_plot()
        #except:
            #print(colored("Not a Valid Coordinate System. Can either be geographic or cartesian","red"))

    def init_plot(self):
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 4])
        self.ax3 = self.fig.add_subplot(gs[0, :])
        self.ax = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax2 = self.fig.add_subplot(gs[1, 1], projection='3d')

        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Altitude (km)')
        self.ax.set_xlim(0, self.x_dim)
        self.ax.set_ylim(0, self.y_dim)
        self.ax.set_zlim(0, self.z_dim)

        self.path_plot, = self.ax.plot([], [], [], color='black')
        self.scatter = self.ax.scatter([], [], [], color='black')
        self.ground_track, = self.ax.plot([], [], [], color='red')
        self.scatter_goal = self.ax.scatter([], [], [], color='green')
        self.canvas = self.fig.canvas

        self.FlowField3D.visualize_3d_planar_flow(self.ax2, skip=self.render_skip)

        self.current_state_line, = self.ax.plot([], [], [], 'r--')

        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius, color='g-')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_inner, color='g--')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_outer, color='g--')

        self.altitude_line, = self.ax3.plot([], [], 'b-')
        self.ax3.set_xlabel('Number of Steps (dt=' + str(self.dt) + ')')
        self.ax3.set_ylabel('Altitude (km)')
        self.ax3.set_xlim(0, self.episode_length)
        self.ax3.set_ylim(0, self.z_dim)

    def init_plot_geographic(self):
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 4])
        self.ax3 = self.fig.add_subplot(gs[0, :])
        self.ax = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax2 = self.fig.add_subplot(gs[1, 1], projection='custom3dquiver')

        self.ax.set_xlabel('X_proj (m)')
        self.ax.set_ylabel('Y_proj (m)')
        self.ax.set_zlabel('Altitude (km)')


        rel_x, rel_y = transform.latlon_to_meters_spherical(self.start_coord["lat"],self.start_coord["lon"], self.start_coord["lat"],self.start_coord["lon"])


        '''
        #Figure out projected coordinates
        x_min1, y_min1 =  self.p(longitude = self.FlowField3D.gfs.LON_LOW, latitude = self.FlowField3D.gfs.LAT_LOW)
        x_min2, y_max1 = self.p(longitude=self.FlowField3D.gfs.LON_LOW, latitude=self.FlowField3D.gfs.LAT_HIGH)

        x_max1, y_min2 = self.p(longitude=self.FlowField3D.gfs.LON_HIGH, latitude=self.FlowField3D.gfs.LAT_LOW)
        x_max2, y_max2 = self.p(longitude=self.FlowField3D.gfs.LON_HIGH, latitude=self.FlowField3D.gfs.LAT_HIGH)


        x_min = round(min(x_min1, x_min2),-5)
        x_max = round(max(x_max1, x_max2), -5)

        y_min = round(min(y_min1, y_min2), -5)
        y_max = round(max(y_max1, y_max2), -5)

        print(x_min,x_max,y_min,y_max)
        '''


        #self.ax.set_xlim(x_min, x_max)
        #self.ax.set_ylim(y_min, y_max)

        self.ax.set_xlim(-150*1000, 150*1000)
        self.ax.set_ylim(-150*1000, 150*1000)
        self.ax.set_zlim(env_params['alt_min'], env_params['alt_max'])

        self.path_plot, = self.ax.plot([], [], [], color='black')
        self.scatter = self.ax.scatter([], [], [], color='black')
        self.ground_track, = self.ax.plot([], [], [], color='red')
        self.scatter_goal = self.ax.scatter([], [], [], color='green')
        self.canvas = self.fig.canvas

        self.FlowField3D.visualize_3d_planar_flow(self.ax2, skip=self.render_skip)

        self.current_state_line, = self.ax.plot([], [], [], 'r--')

        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius, color='g-')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_inner, color='g--')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_outer, color='g--')

        self.altitude_line, = self.ax3.plot([], [], 'b-')
        self.ax3.set_xlabel('Number of Steps (dt=' + str(self.dt) + ')')
        self.ax3.set_ylabel('Altitude (km)')
        self.ax3.set_xlim(0, self.episode_length)
        self.ax3.set_ylim(env_params['alt_min'],env_params['alt_max'])

    def reset(self, goal):
        if hasattr(self, 'fig'):
            plt.close('all')
            delattr(self, 'fig')
            delattr(self, 'ax')
            delattr(self, 'ax2')
            delattr(self, 'ax3')
            delattr(self, 'goal')
            delattr(self, 'scatter')
            delattr(self, 'canvas')

        if self.coordinate_system == "geographic":
            rel_x, rel_y = transform.latlon_to_meters_spherical(self.start_coord["lat"], self.start_coord["lon"],
                                                                self.start_coord["lat"], self.start_coord["lon"])

            self.goal = {"x": rel_x, "y": rel_y}

        if self.coordinate_system == "cartesian":
            self.goal = goal

        self.render_step = 1


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
            z = np.full_like(x, 15000)

        ax.plot(x, y, z, color)

    def render(self, state, path, altitude_history, mode='human'):

        if not hasattr(self, 'fig'):
            if self.coordinate_system == "geographic":
                self.init_plot_geographic()

            if self.coordinate_system == "cartesian":
                self.init_plot()

        if self.render_step == self.render_count:

            self.path_plot.set_data(np.array(path)[:, :2].T)
            self.path_plot.set_3d_properties(np.array(path)[:, 2])

            self.ground_track.set_data(np.array(path)[:, :2].T)
            self.ground_track.set_3d_properties(np.zeros(len(path)))

            self.scatter._offsets3d = (
            np.array([state["x"]]), np.array([state["y"]]), np.array([state["z"]]))
            self.scatter_goal._offsets3d = (np.array([self.goal["x"]]), np.array([self.goal["y"]]), np.array([env_params['alt_min']]))

            self.current_state_line.set_data([state["x"], state["x"]], [state["y"], state["y"]])
            self.current_state_line.set_3d_properties([env_params['alt_min'], state["z"]])

            self.altitude_line.set_data(range(len(altitude_history)), altitude_history)

            self.canvas.draw()
            # self.canvas.flush_events()

            self.render_step = 1

            if mode == 'human':
                plt.pause(0.001)

        else:
            self.render_step += 1