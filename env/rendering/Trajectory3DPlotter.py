import matplotlib.pyplot as plt
import numpy as np
from env.config.env_config import env_params

class Trajectory3DPlotter:
    """
    Handles the 3D visualization of balloon trajectories, including altitude plots and goal radius visualization.

    Attributes:
        ax (matplotlib.axes._axes.Axes): 3D axes object for plotting.
        radius (float): Radius of the target zone in meters.
        radius_inner (float): Inner radius for visualizing proximity to the goal.
        radius_outer (float): Outer radius for visualizing extended proximity.
        goal (dict): Dictionary containing the goal's x and y coordinates.
        dt (float): Time step of the simulation.
        episode_length (int): Total length of the episode in steps.
    """
    def __init__(self, ax, radius, goal, dt, episode_length):
        """
        Initialize the 3D plot elements.

        Args:
            ax (matplotlib.axes._axes.Axes): 3D axes object for plotting.
            radius (float): Radius of the target zone in meters.
            goal (dict): Dictionary containing the goal's x and y coordinates.
            dt (float): Time step of the simulation.
            episode_length (int): Total length of the episode in steps.
        """
        # Initialize 3D plot elements
        self.ax = ax

        self.radius = radius  # m
        self.radius_inner = radius * .5  # m
        self.radius_outer = radius * 1.5  # m

        self.goal = goal
        self.dt = dt
        self.episode_length = episode_length

        self.path_plot, = self.ax.plot([], [], [], color='black')
        self.scatter = self.ax.scatter([], [], [], color='black')
        self.ground_track, = self.ax.plot([], [], [], color='red')
        self.scatter_goal = self.ax.scatter([], [], [], color='green')
        self.current_state_line, = self.ax.plot([], [], [], 'r--')
        self.altitude_line, = self.ax.plot([], [], 'b-')

        self._init_plot()

    def _init_plot(self):
        """
        Initialize the 3D plot
        """
        self.ax.set_xlabel('X_proj (m)')
        self.ax.set_ylabel('Y_proj (m)')
        self.ax.set_zlabel('Altitude (km)')

        self.ax.set_xlim(-env_params['rel_dist'], env_params['rel_dist'])
        self.ax.set_ylim(-env_params['rel_dist'], env_params['rel_dist'])
        self.ax.set_zlim(env_params['alt_min'], env_params['alt_max'])

        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius, color='g-')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_inner, color='g--')
        self.plot_circle(self.ax, self.goal["x"], self.goal["y"], self.radius_outer, color='g--')

    def plot_circle(self, ax, center_x,center_y, radius, plane='xy', color ='g--'):
        """
        Plot a circle at the centered base of the the XY plane

        Args:
            ax (matplotlib.axes._axes.Axes): Axes object for plotting.
            center_x (float): X-coordinate of the circle center.
            center_y (float): Y-coordinate of the circle center.
            radius (float): Radius of the circle.
            plane (str, optional): Plane in which to plot the circle ('xy' by default).
            color (str, optional): Color and style of the circle.
        """
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

    def update(self, Balloon, path, goal):
        """
        Update the trajectory plot with the current state and path.

        Args:
            Balloon (object): Current state of the balloon, including x, y, and altitude.
            path (numpy.ndarray): Array of x, y, altitude coordinates representing the path.
            goal (dict): Dictionary containing the goal's x and y coordinates.
        """
        # Update the trajectory and state plots
        self.path_plot.set_data(np.array(path)[:, :2].T)
        self.path_plot.set_3d_properties(np.array(path)[:, 2])

        self.ground_track.set_data(np.array(path)[:, :2].T)
        self.ground_track.set_3d_properties(np.full(len(path), env_params['alt_min']))

        self.scatter._offsets3d = (
            np.array([Balloon.x]), np.array([Balloon.y]), np.array([Balloon.altitude]))
        self.scatter_goal._offsets3d = (np.array([goal["x"]]), np.array([goal["y"]]), np.array([env_params['alt_min']]))

        self.current_state_line.set_data([Balloon.x, Balloon.x], [Balloon.y, Balloon.y])
        self.current_state_line.set_3d_properties([env_params['alt_min'], Balloon.altitude])

        self.altitude_line.set_data(range(len(path)), path[:, 2])

    def update_altitude(self, path):
        """
        Update the altitude line plot.

        Args:
            path (numpy.ndarray): Array of x, y, altitude coordinates.
        """
        # Update the altitude plot (for the altitude line)
        self.altitude_line.set_data(range(len(path)), path[:, 2])

    def draw(self):
        """
        Redraw the plot after updates.
        """
        # Redraw the plot to update the changes
        self.ax.figure.canvas.draw()

    def clear(self):
        """
        Clear the plot for reinitialization.
        """
        # Clear the plot when needed
        self.ax.clear()
