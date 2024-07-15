import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import xarray as xr
#import ERA5

class FlowField3D:
    def __init__(self, x_dim, y_dim, z_dim, num_levels, min_vel, max_vel, res, seed):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_levels = num_levels
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.res = res


        self.seed(seed)

        self.flow_field, self.directions, self.magnitudes = self.initialize_flow()

    def seed(self, seed=None):
        if seed != None:
            self.np_rng = np.random.default_rng(seed)
        else:
            self.np_rng = np.random.default_rng(np.random.randint(0, 2**32))

    def randomize_flow(self):
        '''
        Generates a flow field with N number of levels choosing from angles of [0, np.pi / 2, np.pi, 3 * np.pi / 2].
        All angles need to appear at least once, but their order and repetition is random.

        '''
        #UPDATE: This code was not randomizing correctly, Occasionally there were flows that didn't include all 4 angles.

        #Create buckets to randomly choose from
        directions_bucket = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        magnitudes_bucket = [self.max_vel]


        directions_result = directions_bucket
        magnitudes_result = magnitudes_bucket

        #Extend and shuffle arrays to ranndomize flow field

        while len(directions_result) < self.num_levels:
            directions_result.append(self.np_rng.choice(directions_bucket))

        while len(magnitudes_result) < self.num_levels:
            #magnitudes_result.append(self.np_rng.choice(magnitudes_bucket))
            magnitudes_result.append(self.np_rng.uniform(self.min_vel, self.max_vel)) #randomize magnitudes


        # Shuffle the arrays to randomize the order
        self.np_rng.shuffle(directions_result)
        self.np_rng.shuffle(magnitudes_result)

        self.directions = directions_result
        self.magnitudes = magnitudes_result

        #Debug text to make sure flow are being "randomly generated" the same accross multiple envs.
        #print("New flows:", self.directions)
        #print("New flows Magnitudes:", self.magnitudes)

        self.generate_random_planar_flow_field()


    def gradualize_random_flow(self, max_angle_change=np.deg2rad(10)):
        """
        Gradually randomizes the flow field over time instead of fully new flow field each episode.

        The flow at each level is altered by a maximum of +/- ANGLE_CHANGE.

        NOTE: THis function doesn't garuntee random flows will have solveable paths for future episodes,
        since the gradual changes are non uniform every episode and can be either positive or negative.
        """
        new_directions = []
        for direction in self.directions:
            angle_change = self.np_rng.uniform(-max_angle_change, max_angle_change)
            new_direction = (direction + angle_change) #% (2 * np.pi)
            new_directions.append(new_direction)

        self.directions = new_directions
        print("New flows:", self.directions)
        self.generate_random_planar_flow_field()

    def initialize_flow(self):
        """
        Technically this function is unecessary now.  However it gives a way to override the random flow function
        and generate a static flow instead of a random flow.

        """
        self.directions = self.np_rng.uniform(0, 2 * np.pi, size=self.num_levels)
        self.magnitudes = self.np_rng.uniform(self.min_vel, self.max_vel, size=self.num_levels)

        # Static Debugging:
        self.directions = [0,  np.pi/2, np.pi, 3*np.pi/2, 0, np.pi]
        self.magnitudes = [5, 10, 5, 10, 10, 5]

        # Static Debugging2:
        #self.directions = [np.pi, 0, np.pi, 3 * np.pi / 2, np.pi / 2, 0]
        #self.magnitudes = [10, 5, 10, 5, 10, 5]

        self.randomize_flow() #Comment this out, to have hard coded static flows.

        return self.generate_random_planar_flow_field()

    # UPDATE new function
    def convert_range(self,value, old_min, old_max, new_min, new_max):
        # Calculate the proportional value
        new_value = new_min + ((value - old_min) / (old_max - old_min)) * (new_max - new_min)
        return new_value

    def interpolate_xr(self, x, y, z):

        # Perform interpolation on x, y, and level dimensions
        interp_ds = self.ds.interp(x=x, y=y, level=self.ds['level'])

        # Now interpolate the result over z
        # Extract interpolated values and the corresponding z-values
        z_interp_values = interp_ds['z'].values
        u_interp_values = interp_ds['u'].values
        v_interp_values = interp_ds['v'].values

        # Create a function to perform interpolation over z
        def interpolate_over_z(z_values, data_values, target_z):
            #print(z_values)
            #print(data_values)
            #print(target_z)
            return griddata(z_values, data_values, target_z, method='linear')

        # Interpolate the data variables over z
        interp_u = interpolate_over_z(z_interp_values, u_interp_values, z)
        interp_v = interpolate_over_z(z_interp_values, v_interp_values, z)

        # Create a new dataset with the interpolated values
        final_ds = xr.Dataset(
            {
                "u": ([], interp_u),
                "v": ([], interp_v)
            },
            coords={
                "x": x,
                "y": y,
                "z": z
            }
        )



        return final_ds['u'].values, final_ds['v'].values

    def interpolate_xr2(self, x, y, z):
        #lev = self.convert_range(z, 0, self.z_dim, 0, self.num_levels)

        #z has already been converted to level

        # Perform interpolation
        #I think this is wrong
        final_ds = self.ds.interp(x=x, y=y, level=z)

        return final_ds['u'].values, final_ds['v'].values



    #time this
    def create_netcdf(self):
        self.ds = xr.Dataset(
            {
                "u": (["level", "x", "y"], self.flow_field[:,:,: ,0]),
                "v": (["level", "x", "y"], self.flow_field[:,:,:, 1]),
                "z": (["level", "x", "y"], self.flow_field[:, :, :, 3]),
            },
            coords={
                "level": np.arange(self.flow_field.shape[0]),
                "x": np.arange(self.flow_field.shape[1]),
                "y": np.arange(self.flow_field.shape[2])
            }
        )

        #Interpolate using EarthSHAB method:

        y = 5 # y (km)
        x = 5 # x (km)
        alt = 2 #km

        #self.ds_np= self.ds.to_numpy()

        #print(type(self.ds_np))
        #print(self.ds_np)



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

        #Update:  dimensions have +1 to include flows on the boundaries.
        flow_field = np.zeros((self.num_levels, self.x_dim+1*self.res, self.y_dim+1*self.res, 4))
        for z in range(self.num_levels):
            u = round(np.cos(self.directions[z]) * self.magnitudes[z],2)
            v = round(np.sin(self.directions[z]) * self.magnitudes[z], 2)
            flow_field[z, :, :, 0] = u
            flow_field[z, :, :, 1] = v
            flow_field[z, :, :, 2] = 0  # Flow is only in the X-Y plane
            flow_field[z, :, :, 3] = alt_levels[z]  # Store altitude in the fourth dimension

        self.flow_field = flow_field

        #UPDATE new stuff for interpolating
        # Initialize REgular Grid Interpolator Variables, so they aren't reassigned with every call to interpolate_flow()
        self.z_space = np.arange(self.flow_field[:, :, :, 0].shape[0])
        self.x_space = np.arange(self.flow_field[:, :, :, 0].shape[1])
        self.y_space = np.arange(self.flow_field[:, :, :, 0].shape[2])
        self.fn_u = RegularGridInterpolator((self.z_space, self.x_space, self.y_space), flow_field[:, :, :, 0])
        self.fn_v = RegularGridInterpolator((self.z_space, self.x_space, self.y_space), flow_field[:, :, :, 1])

        #leave this out for now
        #self.create_netcdf()

        return self.flow_field, self.directions, self.magnitudes

    #UPDATE new function
    def apply_boundary_decay(self, decay_type='linear'):
        '''
        Decay the flow at an altitude level from the source (wall wind is blowing from) to have more realistic flows in
        indoor enviormenment.
        '''
        for z in range(self.num_levels):
            for x in range(self.x_dim+1*self.res):

                decay_factor_x = self._calculate_decay_factor(x, self.x_dim*self.res, decay_type)
                if self.flow_field[z, x, :, 0][0] >= 0:
                    self.flow_field[z, x, :, 0] *= decay_factor_x
                else:
                    self.flow_field[z, x, :, 0] *= 1- decay_factor_x

            for y in range(self.y_dim+1*self.res):
                decay_factor_y = self._calculate_decay_factor(y, self.y_dim*self.res, decay_type)
                if self.flow_field[z, :, y, 1][0] >=0:
                    self.flow_field[z, :, y, 1] *= decay_factor_y
                else:
                    self.flow_field[z, :, y, 1] *= 1-decay_factor_y

    # UPDATE new function
    def _calculate_decay_factor(self, position, dim_length, decay_type):
        '''
        Different Decay Functions.

        Exponential isn't currently working because of the offset for negative directions.

        1- exp  != exp for reversion directions.  Need to fix this bug

        Could add other custom decay functions here to choose from
        '''
        if decay_type == 'linear':
            return 1 - (position / dim_length)
        elif decay_type == 'exponential':
            return np.exp(-position / dim_length)
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")

    def trilinear_interpolate(self, x, y, z):
        '''
        Method written by Georgiy to do a trilinear interpolation without Scipy.

        Does a X% speed up of RegularGridInterpolator.
        :param x:
        :param y:
        :param z:
        :return:
        '''
        x = np.clip(x, 0, self.x_dim)
        y = np.clip(y, 0, self.y_dim)
        z = np.clip(z, 0, self.z_dim)

        z_convert = self.convert_range(z, 0, self.z_dim, 0, self.num_levels - 1)

        x0 = int(np.floor(x))
        x1 = min(x0 + 1, self.x_dim)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, self.y_dim)
        z0 = int(np.floor(z_convert))
        z1 = min(z0 + 1, self.num_levels - 1)

        xd = (x - x0)
        yd = (y - y0)
        zd = (z_convert - z0)

        # Interpolating u
        c00 = self.flow_field[z0, x0, y0, 0] * (1 - xd) + self.flow_field[z0, x1, y0, 0] * xd
        c01 = self.flow_field[z0, x0, y1, 0] * (1 - xd) + self.flow_field[z0, x1, y1, 0] * xd
        c10 = self.flow_field[z1, x0, y0, 0] * (1 - xd) + self.flow_field[z1, x1, y0, 0] * xd
        c11 = self.flow_field[z1, x0, y1, 0] * (1 - xd) + self.flow_field[z1, x1, y1, 0] * xd

        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        u = c0 * (1 - zd) + c1 * zd

        # Interpolating v
        c00 = self.flow_field[z0, x0, y0, 1] * (1 - xd) + self.flow_field[z0, x1, y0, 1] * xd
        c01 = self.flow_field[z0, x0, y1, 1] * (1 - xd) + self.flow_field[z0, x1, y1, 1] * xd
        c10 = self.flow_field[z1, x0, y0, 1] * (1 - xd) + self.flow_field[z1, x1, y0, 1] * xd
        c11 = self.flow_field[z1, x0, y1, 1] * (1 - xd) + self.flow_field[z1, x1, y1, 1] * xd

        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        v = c0 * (1 - zd) + c1 * zd

        return u, v, 0

    def lookup(self):

        #print(self.ds['u'].sel(x=1.9, y=2, level=1, method="nearest"))

        #self.ds['u'].isel(x=1, y=2, level=1)

        #self.flow_field[1,1,2,0]

        #print(self.ds_np)

        print()

    def interpolate_flow(self, x, y, z):
        '''Perform a trilinear interpolation to determine the u and v flow components at a particular point.


        Completely new Interpolation method using SciPy Regular Grid Interpolator.  I believe this method is more correct,
        but it's also signifcantly slower.  Almost 4X.  Maybe we can optimize somehow?
        '''


        #need to convert Z to nearest altitude index, since we don't interpolate the flow vertically in 1 degree resolution, and instead just do number of levels.
        '''
        z = np.clip(z, 0, self.z_dim)
        z_convert =  self.convert_range(z, 0, self.z_dim, 0, self.num_levels-1)

        pts = np.array([[z_convert,
                         np.clip(x,0,self.x_dim),
                         np.clip(y,0,self.y_dim)]])

        #RegularGridInterpolator Functions are initialized in generate_random_planar_flow_field()
        u = self.fn_u(pts)[0]
        v = self.fn_v(pts)[0]

        #u_interp = self.trilinear_interpolation(x, y, z, 0)
        #v_interp = self.trilinear_interpolation(x, y, z, 1)

        #final_ds['u']
        '''

        #u, v = self.interpolate_xr2(np.clip(x,0,self.x_dim), np.clip(y,0,self.y_dim), z_convert)

        u, v, w = self.trilinear_interpolate(x,y,z)

        #print(z, x, y)
        #print("u_scipy", u, "u_xr", u_interp)
        #print("v_scipy", v, "v_xr", v_interp)
        #print()

        return u, v, 0  # Assuming zero vertical flow


    def visualize_3d_planar_flow(self, ax, skip=1, interpolation_point=None):
        '''
        Plot the Flow Field
        '''
        for z in range(self.flow_field.shape[0]):
            #UPDATE added indexing 'ij' which fixed a hidden bug in visualization when flows are not all the same magnitude.
            X, Y = np.meshgrid(np.arange(self.flow_field.shape[1]), np.arange(self.flow_field.shape[2]), indexing='ij')
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
                    ax.quiver(X[i, j]/self.res, Y[i, j]/self.res, Z[i, j], U[i, j], V[i, j], W[i, j], pivot='tail',
                                #For the small arena
                                #length=self.magnitudes[z] * .5/self.x_dim, arrow_length_ratio=1/self.x_dim, color=colors[i, j])
                                #For the big arena
                                #length = self.magnitudes[z] * 1 , arrow_length_ratio = .25, color = colors[i, j])
                                length = self.magnitudes[z] * 100000, arrow_length_ratio = .02, color = colors[i, j])

        if interpolation_point is not None:
            x, y, z = interpolation_point
            u_interp, v_interp, w_interp = self.interpolate_flow(x, y, z)
            ax.quiver(x, y, z, u_interp, v_interp, w_interp, color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Altitude')
        ax.set_xlim(0, (self.x_dim)/self.res)
        ax.set_ylim(0, (self.y_dim)/self.res)
        ax.set_zlim(0, self.z_dim)  # Set the z-axis limit to the maximum altitude

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
        self.z = np.clip(self.z + dz, 0, self.flow_field_3d.z_dim )
        u, v, _ = self.flow_field_3d.interpolate_flow(int(self.x), int(self.y), self.z)

        # Update position based on flow and timestep
        #UPDATE removed -1 from x_dim
        self.x = np.clip(self.x + u * self.dt, 0, self.flow_field_3d.x_dim)
        self.y = np.clip(self.y + v * self.dt, 0, self.flow_field_3d.y_dim)

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

# Example usage
if __name__ == '__main__':
    x_dim = 10
    y_dim = 10
    z_dim = 5
    num_levels = 6
    min_vel = 2
    max_vel = 2
    skip = 1 #for speeding up visualization.  skip ever X quiver.
    res = 1
    seed = None

    #If you uncomment seeding,  the random path generation will be the same every time as long as seed is not None
    #np.random.seed(seed) #seeding

    flow_field_3d = FlowField3D(x_dim, y_dim, z_dim, num_levels, min_vel, max_vel, res, seed)

    # Apply the boundary decay
    #flow_field_3d.apply_boundary_decay(decay_type='linear')

    # Initialize the point mass
    mass = 1.0  # Mass of the point mass
    dt = 0.1  # Time step for simulation
    point_mass = PointMass(flow_field_3d, x=5, y=5, z=0, mass=mass, dt=dt)

    # Simulate the point mass movement
    steps = 60
    dz_func = lambda: np.random.choice([.1])  # Randomly move up, down, or stay at the same altitude
    point_mass.simulate(steps, dz_func)

    # Create a single figure with two subplots
    fig = plt.figure(figsize=(15, 10))

    # Plot the flow field
    ax1 = fig.add_subplot(121, projection='3d')
    flow_field_3d.visualize_3d_planar_flow(ax1, skip, interpolation_point=(250, 250, 25))

    # Plot the point mass path
    ax2 = fig.add_subplot(122, projection='3d')
    point_mass.visualize_path(ax2)

    #Check Interpolation
    print(flow_field_3d.interpolate_flow(0, 0, 0))
    print(flow_field_3d.interpolate_flow(0, 0, 1))

    print(flow_field_3d.interpolate_flow(0, 0, .5))
    print(flow_field_3d.interpolate_flow(4.99,4.99, .5))

plt.show()
