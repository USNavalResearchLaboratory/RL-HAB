import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, sys.path[0] + '/../')
from utils.Custom3DQuiver import Custom3DQuiver
from matplotlib.projections import register_projection

# Register the custom projection
register_projection(Custom3DQuiver)


fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(111, projection='custom3dquiver')


# Manually add a CustomAxes3D to the figure
#ax = Custom3DQuiver(fig)
#fig.add_axes(ax)

# Make data
n = 4
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
z = np.linspace(-10, 10, n)
X, Y, Z = np.meshgrid(x, y, z)
U = (X + Y)/5
V = (Y - X)/5
W = Z*0


ax.quiver(X,Y,Z,U,V,W, length=1, arrow_length_ratio=1, arrow_head_angle=90)



# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)
#ax2 = fig.add_subplot(122, projection='3d')
#ax2.plot(x, -y)

plt.show()
