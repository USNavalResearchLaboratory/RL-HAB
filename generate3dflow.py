import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpn, griddata
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate


class FlowField3D:
    def __init__(self, x_dim, y_dim, z_dim, num_levels, min_vel, max_vel):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_levels = num_levels
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.flow_field, self.directions, self.magnitudes = self.generate_random_planar_flow_field()

    def generate_random_planar_flow_field(self):
        directions = np.random.uniform(0, 2 * np.pi, size=self.num_levels)
        magnitudes = np.random.uniform(self.min_vel, self.max_vel, size=self.num_levels)
        alt_levels = np.linspace(0, self.z_dim, self.num_levels, endpoint=True)

        flow_field = np.zeros((self.num_levels, self.x_dim, self.y_dim, 4))
        for z in range(self.num_levels):
            u = np.cos(directions[z]) * magnitudes[z]
            v = np.sin(directions[z]) * magnitudes[z]
            flow_field[z, :, :, 0] = u
            flow_field[z, :, :, 1] = v
            flow_field[z, :, :, 2] = 0  # Flow is only in the X-Y plane
            flow_field[z, :, :, 3] = alt_levels[z]  # Store altitude in the fourth dimension

        return flow_field, directions, magnitudes

    def interpolate_flow(self, x, y, z):
        alt_levels = self.flow_field[:, 0, 0, 3]
        above_idx = np.searchsorted(alt_levels, z, side='right')
        below_idx = above_idx - 1 if above_idx > 0 else 0

        flow_below = self.flow_field[below_idx, :, :, :3]
        flow_above = self.flow_field[above_idx, :, :, :3]

        weight_below = (alt_levels[above_idx] - z) / (alt_levels[above_idx] - alt_levels[below_idx])
        weight_above = 1.0 - weight_below

        u_interp = weight_below * flow_below[x, y, 0] + weight_above * flow_above[x, y, 0]
        v_interp = weight_below * flow_below[x, y, 1] + weight_above * flow_above[x, y, 1]
        w_interp = 0  # Flow is only in the X-Y plane

        return u_interp, v_interp, w_interp

    def visualize_3d_planar_flow(self, skip=1, interpolation_point=None):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        for z in range(self.flow_field.shape[0]):
            X, Y = np.meshgrid(np.arange(self.flow_field.shape[1]), np.arange(self.flow_field.shape[2]))
            U = self.flow_field[z, :, :, 0]
            V = self.flow_field[z, :, :, 1]
            W = self.flow_field[z, :, :, 2]  # Flow is only in the X-Y plane
            Z = np.full_like(X, self.flow_field[z, 0, 0, 3])

            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], Z[::skip, ::skip],
                      U[::skip, ::skip], V[::skip, ::skip], W[::skip, ::skip],
                      length=self.magnitudes[z] * 1, normalize=True, arrow_length_ratio=.1)

        if interpolation_point is not None:
            x, y, z = interpolation_point
            u_interp, v_interp, w_interp = self.interpolate_flow(x, y, z)
            ax.quiver(x, y, z, u_interp, v_interp, w_interp, color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Altitude')
        ax.set_xlim(0, self.flow_field.shape[1])
        ax.set_ylim(0, self.flow_field.shape[2])
        ax.set_zlim(0, self.flow_field[-1, 0, 0, 3])  # Set the z-axis limit to the maximum altitude

        plt.show()


class PointMass:
    def __init__(self, flow_field_3d, x, y, z, mass, dt):
        self.flow_field_3d = flow_field_3d
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass
        self.dt = dt
        self.path = [(x, y, z)]
        self.velocity = np.array([0.0, 0.0, 0.0])

    def move(self, dz):
        """
        Move the point mass vertically by dz units and let the flow dictate the horizontal movement.

        :param dz: Change in altitude (can be positive or negative).
        """
        self.z = np.clip(self.z + dz, 0, self.flow_field_3d.z_dim - 1)
        u, v, _ = self.flow_field_3d.interpolate_flow(int(self.x), int(self.y), self.z)

        # Update position based on flow and timestep
        self.x = np.clip(self.x + u * self.dt, 0, self.flow_field_3d.x_dim - 1)
        self.y = np.clip(self.y + v * self.dt, 0, self.flow_field_3d.y_dim - 1)

        # Update velocity
        self.velocity = np.array([u, v, 0])

        self.path.append((self.x, self.y, self.z))

    def simulate(self, steps, dz_func):
        """
        Simulate the point mass movement through the flow field for a given number of steps.

        :param steps: Number of steps to simulate.
        :param dz_func: Function to determine dz at each step.
        """
        for _ in range(steps):
            dz = dz_func()
            self.move(dz)

    def visualize_path(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Altitude')
        ax.set_xlim(0, self.flow_field_3d.x_dim)
        ax.set_ylim(0, self.flow_field_3d.y_dim)
        ax.set_zlim(0, self.flow_field_3d.z_dim)

        plt.show()


# Example usage
x_dim = 100
y_dim = 100
z_dim = 100
num_levels = 5
min_vel = 1
max_vel = 10
skip = 10

flow_field_3d = FlowField3D(x_dim, y_dim, z_dim, num_levels, min_vel, max_vel)
flow_field_3d.visualize_3d_planar_flow(skip)

# Initialize the point mass
mass = 1.0  # Mass of the point mass
dt = 1.0  # Time step for simulation
point_mass = PointMass(flow_field_3d, x=50, y=50, z=25, mass=mass, dt=dt)

# Simulate the point mass movement
steps = 50
dz_func = lambda: np.random.choice([-1, 0, 5])  # Randomly move up, down, or stay at the same altitude
point_mass.simulate(steps, dz_func)

# Visualize the path
point_mass.visualize_path()
