import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pyrobot.geom2d import Point

from pyrobot.geom3d import RotMatrix, Vector, TransformMatrix

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

def generate_obb_vertex(self,C,geom):
    '''
    C - DH matrix value
    geom [x,y,z] - half lenght
    '''
    C = np.matrix(C)
    center = self.parent.kin.get_center(C)
    C0 = self.parent.kin.get_versor(C,axis=0)
    C1 = self.parent.kin.get_versor(C,axis=1)
    C2 = self.parent.kin.get_versor(C,axis=2)
    p0 = center + geom[0]*C0 + geom[1]*C1 + geom[2]*C2
    p1 = center + geom[0]*C0 + geom[1]*C1 - geom[2]*C2
    p2 = center + geom[0]*C0 - geom[1]*C1 + geom[2]*C2
    p3 = center + geom[0]*C0 - geom[1]*C1 - geom[2]*C2
    p4 = center - geom[0]*C0 + geom[1]*C1 + geom[2]*C2
    p5 = center - geom[0]*C0 + geom[1]*C1 - geom[2]*C2
    p6 = center - geom[0]*C0 - geom[1]*C1 + geom[2]*C2
    p7 = center - geom[0]*C0 - geom[1]*C1 - geom[2]*C2
    R = np.concatenate((p0,p1,p2,p3,p4,p5,p6,p7))
    i = 0
    x = np.array([
    [p1[0,i],p3[0,i],p7[0,i],p5[0,i],p1[0,i]],
    [p0[0,i],p2[0,i],p6[0,i],p4[0,i],p0[0,i]],
    [p1[0,i],p5[0,i],p4[0,i],p0[0,i],p1[0,i]],
    [p3[0,i],p7[0,i],p6[0,i],p2[0,i],p3[0,i]]
    ])
    i = 1
    y = np.array([
    [p1[0,i],p3[0,i],p7[0,i],p5[0,i],p1[0,i]],
    [p0[0,i],p2[0,i],p6[0,i],p4[0,i],p0[0,i]],
    [p1[0,i],p5[0,i],p4[0,i],p0[0,i],p1[0,i]],
    [p3[0,i],p7[0,i],p6[0,i],p2[0,i],p3[0,i]]
    ])
    i = 2
    z = np.array([
    [p1[0,i],p3[0,i],p7[0,i],p5[0,i],p1[0,i]],
    [p0[0,i],p2[0,i],p6[0,i],p4[0,i],p0[0,i]],
    [p1[0,i],p5[0,i],p4[0,i],p0[0,i],p1[0,i]],
    [p3[0,i],p7[0,i],p6[0,i],p2[0,i],p3[0,i]]
    ])
    return R,x,y,z

def obb(self,C,ax=None,x_length=10,y_length=10,z_length=30,color='b',alpha=0.8):
    C = np.matrix(C)
    if ax is None:
        ax = self.ax
    # OBB
    _,x,y,z = self.generate_obb_vertex(C,[x_length,y_length,z_length])
    ax.plot_surface(x,y,z, color=color, rstride=1, cstride=1, alpha=alpha)
def circle_3d(self,C,ax=None,ax_normal=2,R=10,color='r'):
    C = np.matrix(C)
    if ax is None:
        ax = self.ax
    ax_list = [0,1,2]
    ax_list.remove(ax_normal)
    theta = np.linspace(0, 2 * np.pi, 100)
    center = self.parent.kin.get_center(C,array=True)
    normal_vector = self.parent.kin.get_versor(C,axis=ax_normal,array=True)
    perpendicular_vector = self.parent.kin.get_versor(C,axis=ax_list[0],array=True) #vecotr from circle center to any point on the circumference
    X = []
    Y = []
    Z = []
    for t in theta:
        x,y,z = R*np.cos(t)*perpendicular_vector + R*np.sin(t)*np.cross(normal_vector,perpendicular_vector) + center
        X.append(x)
        Y.append(y)
        Z.append(z)
    ax.plot(X,Y,Z,color=color)
def sphere(self,C,ax=None,R=10,n=50,color='b',alpha=0.8):
    C = np.matrix(C)
    if ax is None:
        ax = self.ax
    center = self.parent.kin.get_center(C,array=True)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = R * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = R * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = R * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, linewidth=0, alpha=alpha)

def add_lcs(self,C,param=None):
    '''
    C = kin.DH_Matrix([1,0,0,0])
    kin.reset_plot()
    kin.add_plot(C,no=1,length=2)
    kin.draw_plot(4)
    '''
    if param is None:
        param = self.Param()
    self.lcs(C,param=param)
    center = self.parent.kin.get_center(C)
    self.last_c.append(center)
    if param.draw_to_element >= 0:
        self.ax.plot([self.last_c[param.draw_to_element][0,0], center[0,0]], 
                     [self.last_c[param.draw_to_element][0,1], center[0,1]], 
                     [self.last_c[param.draw_to_element][0,2], center[0,2]],color=param.draw_color,linewidth=param.draw_width)
def draw_chain_from_table(self,D,param=[],print_flag=False,title=[],fname=[],trajectory_trace=False,draw_flag=True):
    '''
    D = []
    D.append([0,0,0,0])
    D.append([0,1,3,1])
    D.append([1,0,0,0])
    D.append([0,5,0,0])
    draw = [1,0,1,1]
    kin.draw_chain_from_table(D,draw)
    '''
    self.last_c = []
    if len(param) == 0:
        for i,d in enumerate(D):
            param.append(self.Param(no=i))
    C = np.eye(4)
    for i,d in enumerate(D):
        # obliczenie macierzy transformacji
        C = self.parent.kin.dh(d,C)
        # wyświetlenie LCS (LUW) wszystkich lub wskazanych w liście draw_list
        self.add_lcs(C,param=param[i])
        # wyświetlenie cylindra lub prostopadłościanu w zależności od zmiennej złączowej
        length = param[i].lcs_length
        if param[i].joint_marker == 'rz':
            self.cylinder(C,along_axis=2,R=length/3,L=length/8)
        elif param[i].joint_marker == 'rx':
            self.cylinder(C,along_axis=0,R=length/3,L=length/8)
        elif param[i].joint_marker == 'z':
            self.obb(C,x_length=length/10,y_length=length/10,z_length=length/4)
        elif param[i].joint_marker == 'x':
            self.obb(C,x_length=length/4,y_length=length/10,z_length=length/10)
        # opis
        if print_flag:
            print('--- %d ---------------' % i)
            print(d)
            print(C)
    # wyświetlanie dodatkowych punktów trajektorii
    if trajectory_trace:
        center = self.parent.kin.get_center(C,array=True)
        self.trajectory_points[0].append(center[0])
        self.trajectory_points[1].append(center[1])
        self.trajectory_points[2].append(center[2])
        self.ax.plot(self.trajectory_points[0],self.trajectory_points[1],self.trajectory_points[2],color='y',linewidth=2)
    # wyświetlenie układów wsp.
    if draw_flag:
        self.show(title=title,fname=fname)
    # zwrócenie ostatniego punktu
    return self.parent.kin.get_center(C,array=True)
class Param():
    def __init__(self,
                    no=-1,
                    lcs=True,
                    lcs_length=50,
                    lcs_coord=True,
                    joint_marker='none',
                    draw_to_element=-1,
                    draw_color='k',
                    draw_width=1,
                    ax_limits=[],) -> None:
        self.no = no
        self.lcs = lcs # rysowanie lokalnego ukł. wsp.
        self.lcs_length = lcs_length # długość lok. ukł. wsp.
        self.lcs_coord = lcs_coord # wypisywanie wsp. układu
        self.joint_marker = joint_marker # zaznaczanie typu złącza
        self.draw_to_element = draw_to_element # rysowanie połączenie pomiędzy bieżącym ukł. a wskazanym przez podaną wartość
        self.draw_color = draw_color # kolor połączenia
        self.draw_width = draw_width # kolor połączenia
        self.ax_limits = ax_limits # ręczne ustawianie zakresu

def traj(self,R,x=[],y=[],z=[]):
    n = len(R)
    plot_from_matrix_flag = False
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        x=[]
        y=[]
        z=[]
        plot_from_matrix_flag = True
        for r in R:
            tmp = self.parent.kin.get_center(r,array=True)
            x.append(tmp[0])
            y.append(tmp[1])
            z.append(tmp[2])
    self.parent.plot.ax.plot(x,y,z,color='m',linewidth=3)
    if plot_from_matrix_flag:
        self.parent.plot.add_lcs(R[0])
        self.parent.plot.add_lcs(R[-1])
        self.parent.plot.add_lcs(R[round(n/3)])
        self.parent.plot.add_lcs(R[round(2*n/3)])
def joints(self,J,fig=None,pos_profile_flag=True,vel_profile_flag=False,acc_profile_flag=False, dt=0.1):
    if fig is None:
        fig = plt.figure()
    
    splots = 100*len(J[0])+11
    for i in range(0,len(J[0])):
        ax = fig.add_subplot(splots+i)
        pos = []
        for j in J:
            pos.append(j[i])
        vel = self.parent.kin.integrate_table(pos,dt)
        acc = self.parent.kin.integrate_table(vel,dt)
        # vel = []
        # for j,p in enumerate(pos):
        #     if j == 0:
        #         vel.append(0)
        #     else:
        #         vel.append((p-pos[j-1])/dt)
        # if acc_profile_flag:
        #     acc = []
        #     for j,v in enumerate(vel):
        #         if j == 0:
        #             acc.append(0)
        #         else:
        #             acc.append((v-vel[j-1])/dt)
        if pos_profile_flag:
            ax.plot(pos,'b.')
            ax.plot(pos,'b')
        if vel_profile_flag:
            ax.plot(vel,'r.')
            ax.plot(vel,'r')
        if acc_profile_flag:
            ax.plot(acc,'g.')
            ax.plot(acc,'g')
        
        plt.grid()
        title = 'DOF%d' % i
        plt.title(title)
        
    plt.grid()
    # plt.show()

if __name__ == "__main__":

    ax, fig = get_ax_3d()
    C = TransformMatrix()
    C1 = TransformMatrix(rotation=RotMatrix(45,deg=True))
    lcs(ax,C)
    lcs(ax,C1)

    show()