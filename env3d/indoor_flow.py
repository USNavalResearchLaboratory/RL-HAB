import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import matplotlib.colors
import sys
from scipy.interpolate import RegularGridInterpolator

#np.set_printoptions(threshold=sys.maxsize)

class FlowField():

    def __init__(self):
        self.flow_field = self.randomFlowField()


    def addFan(self, x,y,z, orientation, power, step = 1):
        distance_array = np.arange(0, 6.25, step=step)

        #flow calculator,  we'll figure this out later
        linear_flow = (distance_array[::-1]/5 ** .9) + power*1

        vectors = []

        for i in range(0,len(distance_array)):
            vector = [x + distance_array[i] *np.cos(orientation),y + distance_array[i] *np.sin(orientation),z,linear_flow[i]*np.cos(orientation), linear_flow[i]*np.sin(orientation), 0]
            vectors.append(vector)

        return (vectors)


    def addFlowWall(self, type, x, y, z, orientation, num =4):

        vector_wall = []

        if type == 'Y':
            x_range = np.linspace(-4, 4, num = num)
            for i in range(0, num):
                vectors = self.addFan(x=x_range[i], y=y, z=z, orientation=orientation, power=.1, step=1)

                if len(vector_wall) == 0:
                    vector_wall = np.array(vectors)
                else:
                    vector_wall = np.concatenate((vector_wall, vectors), axis=0)
                vector_wall = np.concatenate((vector_wall, vectors), axis=0)

        if type== 'X':
            y_range = np.linspace(-4, 4, num = num)
            for i in range(0, num):
                vectors = self.addFan(x=x, y=y_range[i], z=z, orientation=orientation, power=.1, step=1)
                if len(vector_wall) == 0:
                    vector_wall = np.array(vectors)
                else:
                    vector_wall = np.concatenate((vector_wall, vectors), axis=0)
                vector_wall = np.concatenate((vector_wall, vectors), axis=0)

        return vector_wall


    def randomFlowField(self):
        vector_wall1 = self.addFlowWall(type='X', x=-4, y=None, z=1.5, orientation=0, num=4)
        vector_wall2 = self.addFlowWall(type='Y', x=None, y=4, z=4.5, orientation=-np.pi / 2, num=4)
        vector_wall3 = self.addFlowWall(type='Y', x=None, y=-4, z=2.5, orientation=np.pi / 2, num=4)
        vector_wall4 = self.addFlowWall(type='X', x=4, y=None, z=3.5, orientation=np.pi, num=4)
        vector_wall5 = self.addFlowWall(type='Y', x=None, y=4, z=0.5, orientation=-np.pi / 2, num=4)

        vector_wall6 = self.addFlowWall(type='Y', x=None, y=4, z=0, orientation=-np.pi / 2, num=4)
        vector_wall7 = self.addFlowWall(type='Y', x=None, y=4, z=4.5, orientation=-np.pi / 2, num=4)

        flow_field = np.concatenate((vector_wall1, vector_wall2, vector_wall3, vector_wall4, vector_wall5, vector_wall6, vector_wall7), axis=0)

        return flow_field

    def interpolateFlowCoord(self, coord):
        X, Y, Z, U, V, W = zip(*self.flow_field)

        xx, yy, zz = np.meshgrid(coord[0], coord[1], coord[2])

        points = np.transpose(np.vstack((X, Y, Z)))

        u_interp = interpolate.griddata(points, U, (xx, yy, zz), method='linear', fill_value=np.nan)
        v_interp = interpolate.griddata(points, V, (xx, yy, zz), method='linear', fill_value=np.nan)
        w_interp = interpolate.griddata(points, W, (xx, yy, zz), method='linear', fill_value=np.nan)

        return (u_interp[0][0][0], v_interp[0][0][0], w_interp[0][0][0])

if __name__ == '__main__':

    ff = FlowField()

    X, Y, Z, U, V, W = zip(*ff.flow_field)


    def getColors(V,U):
        # Color by azimuthal angle
        c = np.arctan2(V, U)


        # Convert angles from [-pi,pi]  to [0,pi]
        c = np.mod(c, 2*np.pi)

        # Flatten and normalize  (old method)
        #print(c.min())
        #print(c.ptp())
        #c = (c.ravel() - c.min()) / c.ptp()


        # Normalize angles from 0 to 1
        c = np.interp(c, [0, 2*np.pi], [0,1])

        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 1)))

        # Colormap
        cmap = matplotlib.colormaps["rainbow"]
        #cmap.set_under('k')  # Make masked values transparent
        #cmap.set_clim(-np.pi, np.pi)
        c = cmap(c)

        #c = c.set_bad(alpha=0)
        return c


    x, y, z = (0.5, 0.5, 2.5)
    coord = [x, y, z]

    u, v, w = ff.interpolateFlowCoord(coord)

    print("coord:", coord)
    print("vector:", u, v, w)


    ## Plot Wind Field
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = getColors(V,U)
    q1 = ax.quiver(X, Y, Z, U, V, W, colors = c,cmap='rainbow')
    q2 = ax.quiver(x,y,z,u,v,w)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([0, 5])
    ax.set_title("Non-Interpolated Flow field from Flow Sources")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    q1.set_array(np.linspace(0,2*np.pi,10))
    fig.colorbar(q1)

    ## Plot Interpolated Wind Field


    # Volume shape
    x= np.arange(-4, 4.1, step=.5)
    y= np.arange(-4, 4.1, step=.5)
    z= np.arange(-2, 6, step=.25)


    xx, yy, zz = np.meshgrid(x, y, z)

    points = np.transpose(np.vstack((X, Y, Z)))

    u_interp = interpolate.griddata(points, U, (xx, yy, zz), method='linear', fill_value=np.nan)
    v_interp = interpolate.griddata(points, V, (xx, yy, zz), method='linear', fill_value=np.nan)
    w_interp = interpolate.griddata(points, W, (xx, yy, zz), method='linear', fill_value=np.nan)

    mask = ~np.isnan(u_interp)

    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    u_interp = u_interp[mask]
    v_interp = v_interp[mask]
    w_interp = w_interp[mask]


    c2 = getColors(v_interp.squeeze(),u_interp.squeeze())


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    q= ax2.quiver(xx, yy, zz, u_interp, v_interp, w_interp, colors = c2, cmap='rainbow')

    ax2.set_title("Interpolated Flow Field")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_zlim([0, 5])
    q.set_array(np.linspace(0,2*np.pi,10))
    fig2.colorbar(q)

    plt.show()
