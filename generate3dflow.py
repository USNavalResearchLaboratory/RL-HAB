import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpn, griddata
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate

def generate_random_planar_flow_field(x_dim, y_dim, z_dim, num_levels, min_vel, max_vel):
    """
    Generate a random planar flow field in 3D.

    The shape of the flow field (5, 100, 100, 3) indicates that there are 5 altitude levels, each represented by a 100x100 grid,
    where each grid point has a 3D velocity vector (x, y, z components). This format can represent different flow directions and magnitudes
    at each altitude level in a structured way.

    :param x_dim: The x dimension of the flow field.
    :param y_dim: The y dimension of the flow field.
    :param z_dim: The z dimension of the flow field.
    :param num_levels: The number of altitude levels.
    :param min_vel: The minimum velocity magnitude.
    :param max_vel: The maximum velocity magnitude.
    :return: The generated flow field, directions, and magnitudes.
    """
    directions = np.random.uniform(0, 2*np.pi, size=num_levels)
    magnitudes = np.random.uniform(min_vel, max_vel, size=num_levels)

    alt_levels = np.linspace(0, z_dim, num_levels, endpoint=True)

    flow_field = np.zeros((num_levels, x_dim, y_dim, 4))
    for z in range(num_levels):
        u = np.cos(directions[z]) * magnitudes[z]
        v = np.sin(directions[z]) * magnitudes[z]
        print(z, z_dim)
        flow_field[z, :, :, 0] = u
        flow_field[z, :, :, 1] = v
        flow_field[z, :, :, 2] = 0  # Flow is only in the X-Y plane
        flow_field[z, :, :, 3] = alt_levels[z]  # Store altitude in the fourth dimension

    #print(flow_field[:,:,:,:])
    print(flow_field.shape)

    return flow_field, directions, magnitudes

def interpolate_flow(flow_field, x, y, z):
    """
    Interpolate the flow field at a point in 3D space.

    :param flow_field: The flow field array.
    :param x: The x-coordinate of the point.
    :param y: The y-coordinate of the point.
    :param z: The z-coordinate of the point.
    :return: The interpolated u, v, w components of the flow field at the point.
    """
    # Extract the altitude levels from the flow field
    alt_levels = flow_field[:, 0, 0, 3]

    # Find the nearest altitude level above and below the given z value
    above_idx = np.searchsorted(alt_levels, z, side='right')
    below_idx = above_idx - 1 if above_idx > 0 else 0

    # Get the flow vectors at the nearest altitude levels
    flow_below = flow_field[below_idx, :, :, :3]
    flow_above = flow_field[above_idx, :, :, :3]

    # Calculate the interpolation weights
    weight_below = (alt_levels[above_idx] - z) / (alt_levels[above_idx] - alt_levels[below_idx])
    weight_above = 1.0 - weight_below

    # Interpolate the flow vectors
    u_interp = weight_below * flow_below[x, y, 0] + weight_above * flow_above[x, y, 0]
    v_interp = weight_below * flow_below[x, y, 1] + weight_above * flow_above[x, y, 1]
    w_interp = 0  # Flow is only in the X-Y plane

    return u_interp, v_interp, w_interp


def visualize_3d_planar_flow(flow_field, directions, magnitudes, skip=1, interpolation_point=None):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for z in range(flow_field.shape[0]):
        X, Y = np.meshgrid(np.arange(flow_field.shape[1]), np.arange(flow_field.shape[2]))
        U = flow_field[z, :, :, 0]
        V = flow_field[z, :, :, 1]
        W = flow_field[z, :, :, 2]  # Flow is only in the X-Y plane

        # Set the altitude of the quivers at each level
        Z = np.full_like(X, flow_field[z, 0, 0, 3])

        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], Z[::skip, ::skip],
                  U[::skip, ::skip], V[::skip, ::skip], W[::skip, ::skip],
                  length=magnitudes[z]*1, normalize=True, arrow_length_ratio=.1)

    # Plot the interpolated vector if specified
    if interpolation_point is not None:
        x, y, z = interpolation_point
        u_interp, v_interp, w_interp = interpolate_flow(flow_field, x, y, z)
        ax.quiver(x, y, z, u_interp, v_interp, w_interp, color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altitude')
    ax.set_xlim(0, flow_field.shape[1])
    ax.set_ylim(0, flow_field.shape[2])
    ax.set_zlim(0, flow_field[-1, 0, 0, 3])  # Set the z-axis limit to the maximum altitude

    plt.show()
# Example usage
x_dim = 100
y_dim = 100
z_dim = 100
num_levels = 5
min_vel = 1
max_vel = 10
skip = 10

flow_field, directions, magnitudes = generate_random_planar_flow_field(x_dim, y_dim, z_dim, num_levels, min_vel, max_vel)
visualize_3d_planar_flow(flow_field, directions, magnitudes, skip, interpolation_point=(55, 52, 50))


# Example usage
x = 50
y = 50
z = 25  # Halfway between altitude levels 0 and 50

u_interp, v_interp, w_interp = interpolate_flow(flow_field, x, y, z)
print(flow_field[1,0,0,:])
print(f"Interpolated flow at point ({x}, {y}, {z}): u = {u_interp}, v = {v_interp}, w = {w_interp}")

plt.show()