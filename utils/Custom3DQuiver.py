import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as OriginalAxes3D
import numpy as np
import math
from mpl_toolkits.mplot3d import art3d
from matplotlib.projections import register_projection


class Custom3DQuiver(OriginalAxes3D):
    name = 'custom3dquiver'

    def quiver(self, X, Y, Z, U, V, W, *,
               length=1, arrow_length_ratio=.3, pivot='tail', normalize=False,
               arrow_head_angle=15,
               **kwargs):
        def custom_calc_arrows(UVW):
            #print("Custom calc_arrows function called")
            x = UVW[:, 0]
            y = UVW[:, 1]
            norm = np.linalg.norm(UVW[:, :2], axis=1)
            x_p = np.divide(y, norm, where=norm != 0, out=np.zeros_like(x))
            y_p = np.divide(-x, norm, where=norm != 0, out=np.ones_like(x))
            rangle = math.radians(arrow_head_angle)
            c = math.cos(rangle)
            s = math.sin(rangle)
            r13 = y_p * s
            r32 = x_p * s
            r12 = x_p * y_p * (1 - c)
            Rpos = np.array(
                [[c + (x_p ** 2) * (1 - c), r12, r13],
                 [r12, c + (y_p ** 2) * (1 - c), -r32],
                 [-r13, r32, np.full_like(x_p, c)]])
            Rneg = Rpos.copy()
            Rneg[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1
            Rpos_vecs = np.einsum("ij...,...j->...i", Rpos, UVW)
            Rneg_vecs = np.einsum("ij...,...j->...i", Rneg, UVW)
            return np.stack([Rpos_vecs, Rneg_vecs], axis=1)

        had_data = self.has_data()

        input_args = np.broadcast_arrays(X, Y, Z, U, V, W)
        input_args = [np.ravel(arg) for arg in input_args]
        X, Y, Z, U, V, W = input_args

        if any(len(v) == 0 for v in [X, Y, Z, U, V, W]):
            linec = art3d.Line3DCollection([], **kwargs)
            self.add_collection(linec)
            return linec

        shaft_dt = np.array([0., length], dtype=float)
        arrow_dt = shaft_dt * arrow_length_ratio

        if pivot == 'tail':
            shaft_dt -= length
        elif pivot == 'middle':
            shaft_dt -= length / 2

        XYZ = np.column_stack((X, Y, Z))
        UVW = np.column_stack((U, V, W)).astype(float)

        if normalize:
            norm = np.linalg.norm(UVW, axis=1)
            norm[norm == 0] = 1
            UVW = UVW / norm[:, None]

        if len(XYZ) > 0:
            shafts = (XYZ - np.multiply.outer(shaft_dt, UVW)).swapaxes(0, 1)
            head_dirs = custom_calc_arrows(UVW)
            heads = shafts[:, :1] - np.multiply.outer(arrow_dt, head_dirs)
            heads = heads.reshape((len(arrow_dt), -1, 3))
            heads = heads.swapaxes(0, 1)
            lines = [*shafts, *heads[::2], *heads[1::2]]
        else:
            lines = []

        linec = art3d.Line3DCollection(lines, **kwargs)
        self.add_collection(linec)

        self.auto_scale_xyz(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], had_data)

        return linec

'''
# Register the custom projection
register_projection(Custom3DQuiver)

# Create a figure
fig = plt.figure()

# Add subplots using standard subplot notation
ax1 = fig.add_subplot(111, projection='custom3dquiver')
#ax2 = fig.add_subplot(122)

# Create some data for the 3D quiver plot
x, y, z = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))
u = np.sin(x)
v = np.cos(y)
w = np.ones_like(z)

# Plot using the custom quiver function
ax1.quiver(x, y, z, u, v, w, arrow_head_angle=90)

# Create some data for the 2D line plot
t = np.linspace(0, 2 * np.pi, 100)
s = np.sin(t)

# Plot the 2D line plot
#ax2.plot(t, s)

# Show the plot
plt.show()
'''