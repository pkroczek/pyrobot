import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from pyrobot.geom3d import RotMatrix, Point, TransformMatrix

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)#renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def get_ax_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax, fig

def get_ax_2d():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return ax, fig

def reset_limits(ax):
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])

def update_limits(ax,limits,equal=True):
    lim = ax.get_xlim()
    slim = [lim[0],lim[1]]
    if limits['x'][0] < lim[0]:
        slim[0] = limits['x'][0]
    if limits['x'][1] > lim[1]:
        slim[1] = limits['x'][1]
    x = slim

    lim = ax.get_ylim()
    slim = [lim[0],lim[1]]
    if limits['y'][0] < lim[0]:
        slim[0] = limits['y'][0]
    if limits['y'][1] > lim[1]:
        slim[1] = limits['y'][1]
    y = slim

    lim = ax.get_zlim()
    slim = [lim[0],lim[1]]
    if limits['z'][0] < lim[0]:
        slim[0] = limits['z'][0]
    if limits['z'][1] > lim[1]:
        slim[1] = limits['z'][1]
    z = slim
    
    if equal:
        lim = [min([x[0],y[0],z[0]]), max([x[1],y[1],z[1]])]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
    else:
        ax.set_xlim(x)
        ax.set_ylim(y)
        ax.set_zlim(z)

def show():
    plt.grid()
    plt.show()

def title(title):
    plt.title(title)


def lcs(ax, C :TransformMatrix=TransformMatrix(),no=None,len=1,coord=True):
    '''
    lcs - local righthand coordinate system
    '''

    # get lcs vecor points
    center =  C.get_trans()
    C0 = C.get_versor(axis='x')*len + center
    C1 = C.get_versor(axis='y')*len + center
    C2 = C.get_versor(axis='z')*len + center
    # C0 *= len
    # C1 *= len
    # C2 *= len
    # C0 += center
    # C1 += center
    # C2 += center
    if True:
        # Here we create the arrows:
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        a = Arrow3D([center.x, C0.x], [center.y, C0.y], [center.z, C0.z], **arrow_prop_dict, color='r')
        ax.add_artist(a)
        a = Arrow3D([center.x, C1.x], [center.y, C1.y], [center.z, C1.z], **arrow_prop_dict, color='g')
        ax.add_artist(a)
        a = Arrow3D([center.x, C2.x], [center.y, C2.y], [center.z, C2.z], **arrow_prop_dict, color='b')
        ax.add_artist(a)
        # Give them a name:
        if no is None:
            ax.text(C0.x+0.1, C0.y, C0.z, r'$x$')
            ax.text(C1.x,C1.y+0.1, C1.z, r'$y$')
            ax.text(C2.x,C2.y, C2.z+0.1, r'$z$')
        else:
            ax.text(C0.x+0.1, C0.y, C0.z, f'x{no:d}')
            ax.text(C1.x,C1.y+0.1, C1.z,  f'y{no:d}')
            ax.text(C2.x,C2.y, C2.z+0.1,  f'z{no:d}')
    if coord:
        ax.text(center.x, center.y, center.z-0.1, (f'({center.x:.2f},{center.y:.2f},{center.z:.2f})'))
    
    limits = {  'x':[center.x-len,center.x+len],
                'y':[center.y-len,center.y+len],
                'z':[center.z-len,center.z+len]}
    update_limits(ax,limits)

def line(ax, A :Point, B :Point, linewidth=1, color='b'):
    ax.plot([A.x,B.x],[A.y,B.y],[A.z,B.z],color=color,linewidth=linewidth)

def cylinder(ax,C :TransformMatrix=TransformMatrix(),along_axis='z',R=5,L=10,n=20,color='b',alpha=0.8):
    
    ax_list = ['x','y','z']
    ax_list.remove(along_axis)
    # Cylinder
    center =   C.get_trans()
    along_ax = C.get_versor(axis=along_axis)
    perp1_ax = C.get_versor(axis=ax_list[0])
    perp2_ax = C.get_versor(axis=ax_list[1])

    t = np.linspace(0, L, n)
    theta = np.linspace(0, 2 * np.pi, n)
    t, theta = np.meshgrid(t, theta)
    X, Y, Z = [center[0,i] + along_ax[0,i] * t + R * np.sin(theta) * perp1_ax[0,i] + R * np.cos(theta) * perp2_ax[0,i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

def cone(ax,C :TransformMatrix=TransformMatrix(),along_axis='z',r=1,R=5,L=10,n=20,color='b',alpha=0.8):
    
    ax_list = ['x','y','z']
    ax_list.remove(along_axis)
    # Stożek
    center =   C.get_trans()
    along_ax = C.get_versor(axis=along_axis)
    perp1_ax = C.get_versor(axis=ax_list[0])
    perp2_ax = C.get_versor(axis=ax_list[1])
    t0 = np.linspace(0, L, n)
    R = np.linspace(r,R,n)
    theta = np.linspace(0, 2 * np.pi, n)
    t, theta = np.meshgrid(t0, theta)
    # _, R = np.meshgrid(t0, R)
    X, Y, Z = [center[0,i] + along_ax[0,i] * t + R * np.sin(theta) * perp1_ax[0,i] + R * np.cos(theta) * perp2_ax[0,i] for i in [0, 1, 2]]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

# def generate_obb_vertex(self,C,geom):
#     '''
#     C - DH matrix value
#     geom [x,y,z] - half lenght
#     '''
#     C = np.matrix(C)
#     center = self.parent.kin.get_center(C)
#     C0 = self.parent.kin.get_versor(C,axis=0)
#     C1 = self.parent.kin.get_versor(C,axis=1)
#     C2 = self.parent.kin.get_versor(C,axis=2)
#     p0 = center + geom[0]*C0 + geom[1]*C1 + geom[2]*C2
#     p1 = center + geom[0]*C0 + geom[1]*C1 - geom[2]*C2
#     p2 = center + geom[0]*C0 - geom[1]*C1 + geom[2]*C2
#     p3 = center + geom[0]*C0 - geom[1]*C1 - geom[2]*C2
#     p4 = center - geom[0]*C0 + geom[1]*C1 + geom[2]*C2
#     p5 = center - geom[0]*C0 + geom[1]*C1 - geom[2]*C2
#     p6 = center - geom[0]*C0 - geom[1]*C1 + geom[2]*C2
#     p7 = center - geom[0]*C0 - geom[1]*C1 - geom[2]*C2
#     R = np.concatenate((p0,p1,p2,p3,p4,p5,p6,p7))
#     i = 0
#     x = np.array([
#     [p1[0,i],p3[0,i],p7[0,i],p5[0,i],p1[0,i]],
#     [p0[0,i],p2[0,i],p6[0,i],p4[0,i],p0[0,i]],
#     [p1[0,i],p5[0,i],p4[0,i],p0[0,i],p1[0,i]],
#     [p3[0,i],p7[0,i],p6[0,i],p2[0,i],p3[0,i]]
#     ])
#     i = 1
#     y = np.array([
#     [p1[0,i],p3[0,i],p7[0,i],p5[0,i],p1[0,i]],
#     [p0[0,i],p2[0,i],p6[0,i],p4[0,i],p0[0,i]],
#     [p1[0,i],p5[0,i],p4[0,i],p0[0,i],p1[0,i]],
#     [p3[0,i],p7[0,i],p6[0,i],p2[0,i],p3[0,i]]
#     ])
#     i = 2
#     z = np.array([
#     [p1[0,i],p3[0,i],p7[0,i],p5[0,i],p1[0,i]],
#     [p0[0,i],p2[0,i],p6[0,i],p4[0,i],p0[0,i]],
#     [p1[0,i],p5[0,i],p4[0,i],p0[0,i],p1[0,i]],
#     [p3[0,i],p7[0,i],p6[0,i],p2[0,i],p3[0,i]]
#     ])
#     return R,x,y,z
#
# def obb(self,C,ax=None,x_length=10,y_length=10,z_length=30,color='b',alpha=0.8):
#     C = np.matrix(C)
#     if ax is None:
#         ax = self.ax
#     # OBB
#     _,x,y,z = self.generate_obb_vertex(C,[x_length,y_length,z_length])
#     ax.plot_surface(x,y,z, color=color, rstride=1, cstride=1, alpha=alpha)
# def circle_3d(self,C,ax=None,ax_normal=2,R=10,color='r'):
#     C = np.matrix(C)
#     if ax is None:
#         ax = self.ax
#     ax_list = [0,1,2]
#     ax_list.remove(ax_normal)
#     theta = np.linspace(0, 2 * np.pi, 100)
#     center = self.parent.kin.get_center(C,array=True)
#     normal_vector = self.parent.kin.get_versor(C,axis=ax_normal,array=True)
#     perpendicular_vector = self.parent.kin.get_versor(C,axis=ax_list[0],array=True) #vecotr from circle center to any point on the circumference
#     X = []
#     Y = []
#     Z = []
#     for t in theta:
#         x,y,z = R*np.cos(t)*perpendicular_vector + R*np.sin(t)*np.cross(normal_vector,perpendicular_vector) + center
#         X.append(x)
#         Y.append(y)
#         Z.append(z)
#     ax.plot(X,Y,Z,color=color)
# def sphere(self,C,ax=None,R=10,n=50,color='b',alpha=0.8):
#     C = np.matrix(C)
#     if ax is None:
#         ax = self.ax
#     center = self.parent.kin.get_center(C,array=True)
#     u = np.linspace(0, 2 * np.pi, n)
#     v = np.linspace(0, np.pi, n)
#     x = R * np.outer(np.cos(u), np.sin(v)) + center[0]
#     y = R * np.outer(np.sin(u), np.sin(v)) + center[1]
#     z = R * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
#     ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=0, alpha=alpha)

if __name__ == "__main__":

    ax, fig = get_ax_3d()
    C = TransformMatrix()
    C1 = TransformMatrix(rotation=RotMatrix(45,deg=True))
    lcs(ax,C)
    lcs(ax,C1)

    show()