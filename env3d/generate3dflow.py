import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class FlowField3D:
    def __init__(self, x_dim, y_dim, z_dim, num_levels, min_vel, max_vel, seed):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_levels = num_levels
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.flow_field, self.directions, self.magnitudes = self.initialize_flow()
        np.random.seed(seed)

    def randomize_flow(self):

        directions_bucket = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        magnitudes_bucket = [5]


        # Create the result arrays with at least one occurrence of each value
        directions_result = np.random.choice(directions_bucket, 6, replace=True).tolist()
        magnitudes_result = np.random.choice(magnitudes_bucket, 6, replace=True).tolist()

        # Ensure each value from the original arrays appears at least once
        for value in directions_bucket:
            if value not in directions_result:
                directions_result[np.random.randint(6)] = value
        for value in  magnitudes_bucket:
            if value not in magnitudes_result:
                magnitudes_result[np.random.randint(6)] = value

        # Shuffle the arrays to randomize the order
        np.random.shuffle(directions_result)
        np.random.shuffle(magnitudes_result)

        self.directions = directions_result
        self.magnitudes = magnitudes_result

        print(self.directions)
        print(self.magnitudes)

        self.generate_random_planar_flow_field()


    def gradualize_random_flow(self, max_angle_change=np.deg2rad(10)):
        #print(max_angle_change, self.directions)

        new_directions = []
        for direction in self.directions:
            angle_change = np.random.uniform(-max_angle_change, max_angle_change)
            new_direction = (direction + angle_change) #% (2 * np.pi)
            new_directions.append(new_direction)

        self.directions = new_directions
        self.generate_random_planar_flow_field()


    def initialize_flow(self):
        self.directions = np.random.uniform(0, 2 * np.pi, size=self.num_levels)
        self.magnitudes = np.random.uniform(self.min_vel, self.max_vel, size=self.num_levels)

        # Static Debugging:
        #self.directions = [0,  np.pi/2, np.pi, 3*np.pi/2, 0, np.pi]
        #self.magnitudes = [5, 10, 5, 10, 10, 5]

        # Static Debugging2:
        self.directions = [np.pi, 0, np.pi, 3 * np.pi / 2, np.pi / 2, 0]
        self.magnitudes = [10, 5, 10, 5, 10, 5]

        self.randomize_flow()

        return self.generate_random_planar_flow_field()


    def generate_random_planar_flow_field(self):
        """
        The shape of the flow field is (num_levels, x_dim, y_dim, 4) where x y and z are index values:

            1. flow_field[z, x, y, 0]: The horizontal flow component in the X direction (u).
            2. flow_field[z, x, y, 1]: The horizontal flow component in the Y direction (v).
            3. flow_field[z, x, y, 2]: The vertical flow component, which is set to 0 since the flow is only in the X-Y plane (w).
            4. flow_field[z, x, y, 3]: The altitude level corresponding to the Z dimension.

        :return:
        """
        alt_levels = np.linspace(0, self.z_dim, self.num_levels, endpoint=True)

        #print(alt_levels)

        flow_field = np.zeros((self.num_levels, self.x_dim, self.y_dim, 4))
        for z in range(self.num_levels):
            u = round(np.cos(self.directions[z]) * self.magnitudes[z],2)
            v = round(np.sin(self.directions[z]) * self.magnitudes[z], 2)
            flow_field[z, :, :, 0] = u
            flow_field[z, :, :, 1] = v
            flow_field[z, :, :, 2] = 0  # Flow is only in the X-Y plane
            flow_field[z, :, :, 3] = alt_levels[z]  # Store altitude in the fourth dimension

        self.flow_field = flow_field

        return self.flow_field, self.directions, self.magnitudes

    def interpolate_flow(self, x, y, z):
        # Determine the indices of the levels below and above the current altitude
        below_idx = int(np.floor(z / (self.z_dim / self.num_levels)))
        above_idx = int(np.ceil(z / (self.z_dim / self.num_levels)))

        # Ensure the indices are within bounds
        below_idx = max(0, min(self.num_levels - 1, below_idx))
        above_idx = max(0, min(self.num_levels - 1, above_idx))

        if below_idx == above_idx:
            flow_below = self.flow_field[below_idx, :, :, :3]
            flow_above = flow_below
        else:
            flow_below = self.flow_field[below_idx, :, :, :3]
            flow_above = self.flow_field[above_idx, :, :, :3]

        # Interpolate the flow at the current x, y, and z positions
        frac = (z % (self.z_dim / self.num_levels)) / (self.z_dim / self.num_levels)

        #Clip ranges, so that the agent can go out of bounds, taking the nearest value
        u_below = flow_below[np.clip(int(y), 0, int(self.y_dim)-1), np.clip(int(x), 0, int(self.x_dim)-1), 0]
        v_below = flow_below[np.clip(int(y), 0, int(self.y_dim)-1), np.clip(int(x), 0, int(self.x_dim)-1), 1]
        u_above = flow_above[np.clip(int(y), 0, int(self.y_dim)-1), np.clip(int(x), 0, int(self.x_dim)-1), 0]
        v_above = flow_above[np.clip(int(y), 0, int(self.y_dim)-1), np.clip(int(x), 0, int(self.x_dim)-1), 1]

        u = u_below + frac * (u_above - u_below)
        v = v_below + frac * (v_above - v_below)

        return u, v, 0  # Assuming zero vertical flow


    def visualize_3d_planar_flow(self, ax, skip=1, interpolation_point=None):
        for z in range(self.flow_field.shape[0]):
            X, Y = np.meshgrid(np.arange(self.flow_field.shape[1]), np.arange(self.flow_field.shape[2]))
            U = self.flow_field[z, :, :, 0]
            V = self.flow_field[z, :, :, 1]
            W = self.flow_field[z, :, :, 2]  # Flow is only in the X-Y plane
            Z = np.full_like(X, self.flow_field[z, 0, 0, 3])

            # Calculate directions for color mapping
            directions = np.arctan2(V, U)
            norm = plt.Normalize(-np.pi, np.pi)
            colors = cm.rainbow(norm(directions))

            for i in range(0, X.shape[0], skip):
                for j in range(0, Y.shape[1], skip):
                    ax.quiver(X[i, j], Y[i, j], Z[i, j], U[i, j], V[i, j], W[i, j],
                              length=self.magnitudes[z] * 2, normalize=True, arrow_length_ratio=.5, color=colors[i, j])

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

    def visualize_path(self, ax):
        path = np.array(self.path)
        # Calculate directions for color mapping
        directions = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))
        norm = plt.Normalize(-np.pi, np.pi)
        colors = cm.rainbow(norm(directions))

        for i in range(len(directions)):
            ax.plot(path[i:i+2, 0], path[i:i+2, 1], path[i:i+2, 2], color=colors[i], marker='.')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Altitude')
        ax.set_xlim(0, self.flow_field_3d.x_dim)
        ax.set_ylim(0, self.flow_field_3d.y_dim)
        ax.set_zlim(0, self.flow_field_3d.z_dim)


if __name__ == '__main__':

    # Example usage
    x_dim = 500
    y_dim = 500
    z_dim = 100
    num_levels = 6
    min_vel = 1
    max_vel = 10
    skip = x_dim // 10
    seed = 0
    np.random.seed(seed)

    flow_field_3d = FlowField3D(x_dim, y_dim, z_dim, num_levels, min_vel, max_vel, seed)

    # Initialize the point mass
    mass = 1.0  # Mass of the point mass
    dt = 1.0  # Time step for simulation
    point_mass = PointMass(flow_field_3d, x=250, y=250, z=25, mass=mass, dt=dt)

    # Simulate the point mass movement
    steps = 200
    dz_func = lambda: np.random.choice([-5, -2, 0, 2, 5])  # Randomly move up, down, or stay at the same altitude
    point_mass.simulate(steps, dz_func)

    # Create a single figure with two subplots
    fig = plt.figure(figsize=(15, 10))

    # Plot the flow field
    ax1 = fig.add_subplot(121, projection='3d')
    flow_field_3d.visualize_3d_planar_flow(ax1, skip, interpolation_point=(250, 250, 25))

    # Plot the point mass path
    ax2 = fig.add_subplot(122, projection='3d')
    point_mass.visualize_path(ax2)

plt.show()
