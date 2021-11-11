from dataclasses import dataclass
import numpy as np

# from pyrobot import plot

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)#renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


FACTOR_DEG_2_RAD = (np.pi/180)
FACTOR_RAD_2_DEG = (180/np.pi)

@dataclass
class Point():

    x :float = 0
    y :float = 0
    z :float = 0

    def __str__(self) -> str:
        return f'[x:{self.x :.4f}, y:{self.y :.4f}, z:{self.z :.4f}]'

    def __repr__(self) -> str:
        return self.__str__()

    def to_array(self):
        return np.array([self.x,self.y,self.z])

    def norm(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

@dataclass
class Line():

    a :float = 0
    b :float = 0
    c :float = 0
    p0 : Point = Point()

    def __str__(self) -> str:
        return f'[a:{self.a :.4f}, b:{self.b :.4f}, c:{self.c :.4f}, p0: {self.p0}]'

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_points(self, p0 :Point, p1: Point):
        a = p1.x - p0.x
        b = p1.y - p0.y
        c = p1.z - p0.z
        return Line(a,b,c,p0)

    def get_point_at(self, t :float=0):
        return Point(self.p0.x + self.a*t,self.p0.y + self.b*t,self.p0.z + self.c*t)

@dataclass
class Sphere():

    x :float = 0
    y :float = 0
    z :float = 0
    r :float = 0

    def __str__(self) -> str:
        return f'[x:{self.x :.4f}, y:{self.y :.4f}, z:{self.z :.4f}, r:{self.r :.4f}]'

    def __repr__(self) -> str:
        return self.__str__()

    def to_array(self):
        return np.array([self.x,self.y,self.z,self.r])

    def to_point(self):
        return Point(self.x,self.y,self.z)

@dataclass
class Plane():

    A :float = 0
    B :float = 0
    C :float = 0
    D :float = 0

    def __str__(self) -> str:
        return f'[A:{self.A :.4f},B:{self.B :.4f},C:{self.C :.4f},D:{self.D :.4f}]'

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_points(self,p0 :Point, p1 :Point, p2 :Point):
        self.A = p1.y*p2.z - p2.y*p1.z - p0.y*(p2.z - p1.z) + p0.z*(p2.y - p1.y)
        self.B = p0.x*(p2.z - p1.z ) - (p1.x*p2.z - p2.x*p1.z) + p0.z*(p1.x - p2.x)
        self.C = p0.x*(p1.y - p2.y) - p0.y*(p1.x - p2.x) + (p1.x*p2.y - p2.x*p1.y)
        self.D = -p0.x*(p1.y*p2.z - p2.y*p1.z) + p0.y*(p1.x*p2.z - p2.x*p1.z) - p0.z*(p1.x*p2.y-p2.x*p1.y)

    def to_array(self):
        return np.array([self.A,self.B,self.C,self.D])

@dataclass
class Circle(Sphere):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def __str__(self) -> str:
        return f'[x:{self.x :.4f},y:{self.y :.4f},z:{self.z :.4f},r:{self.r :.4f}]'

    def __repr__(self) -> str:
        return self.__str__()

@dataclass
class Vector(Point):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def get_versor(self):
        norm = self.norm()
        return Vector(self.x/norm,self.y/norm,self.z/norm)

    def __add__(self,other):
        return Vector(self.x+other.x,self.y+other.y,self.z+other.z)

    def __sub__(self,other):
        return Vector(self.x-other.x,self.y-other.y,self.z-other.z)        

    def __mul__(self,other):
        if type(other) == float or type(other) == int:
            return Vector(self.x*other,self.y*other,self.z*other)        
        else:
            raise ValueError('wrong data type')

@dataclass
class RotMatrix():

    R :np.matrix

    def __init__(self,th :float=0,axis :str='none',deg=False,R:np.matrix=None) -> None:

        if R is not None and R.shape == (3,3):
            self.R = R
        else:
            rot = {'x':self.rotX,'y':self.rotY,'z':self.rotZ,'none':self.eye}
            if deg:
                self.R = rot[axis](th*FACTOR_DEG_2_RAD)
            else:
                self.R = rot[axis](th)

    def __str__(self) -> str:
        t  =   f'[{self.R[0,0]:.4f}, {self.R[0,1]:.4f}, {self.R[0,2]:.4f}]'
        t += f'\n[{self.R[1,0]:.4f}, {self.R[1,1]:.4f}, {self.R[1,2]:.4f}]'
        t += f'\n[{self.R[2,0]:.4f}, {self.R[2,1]:.4f}, {self.R[2,2]:.4f}]'
        return t

    def __mul__(self,other):
        R =  self.R * other.R
        return RotMatrix(R=R)

    def get_versor(self,axis='z'):
        d = {'x':0,'y':1,'z':2}
        V = self.R.A[:,d[axis]]
        return Vector(V[0],V[1],V[2])

    def to_zyz(self):
        B = np.arccos(self.R[2,2])
        A = np.arctan2(self.R[1,2],self.R[0,2])
        C = np.arctan2(self.R[2,1],-self.R[2,0])
        return EulerAngle(A,B,C)

    def rotX(self,th):
        r = np.matrix( [[1, 0, 0],
                        [ 0, np.cos(th), -np.sin(th)],
                        [ 0, np.sin(th),  np.cos(th)]] )
        return r

    def rotY(self,th):
        r = np.matrix( [[np.cos(th), 0, np.sin(th)],
                        [0,1,0],
                        [-np.sin(th), 0, np.cos(th)]] )
        return r

    def rotZ(self,th):
        r = np.matrix( [[np.cos(th), -np.sin(th), 0],
                        [np.sin(th), np.cos(th), 0],
                        [0,0,1]] )
        return r

    def eye(self,th):
        r = np.matrix(np.eye(3))
        return r

@dataclass
class TransformMatrix():

    C :np.matrix

    def __init__(self,translation :Vector=Vector(), rotation :RotMatrix=RotMatrix(), C :np.matrix=None) -> None:

        if C is not None and C.shape == (4,4):
            self.C = C
        else:
            C = np.column_stack([rotation.R.A,[translation.x,translation.y,translation.z]])
            self.C = np.matrix(np.vstack([C,[0,0,0,1]]))

    def __str__(self) -> str:
        t  = f'[{self.C[0,0]:.4f}, {self.C[0,1]:.4f}, {self.C[0,2]:.4f}, {self.C[0,3]:.4f}]'
        t += f'\n[{self.C[1,0]:.4f}, {self.C[1,1]:.4f}, {self.C[1,2]:.4f}, {self.C[1,3]:.4f}]'
        t += f'\n[{self.C[2,0]:.4f}, {self.C[2,1]:.4f}, {self.C[2,2]:.4f}, {self.C[2,3]:.4f}]'
        t += f'\n[{self.C[3,0]:.4f}, {self.C[3,1]:.4f}, {self.C[3,2]:.4f}, {self.C[3,3]:.4f}]'
        return t

    def __mul__(self,other):
        C = self.C * other.C
        return TransformMatrix(C=C)

    def show(self):
        xv = self.get_rot().get_versor(axis='x')
        yv = self.get_rot().get_versor(axis='y')
        zv = self.get_rot().get_versor(axis='z')
        pos = self.get_trans()
        print('')
        print(f'POSITION:')
        print(f'\tpos = {pos}')
        print('ORIENTATION:')
        print(f'\tx_versor = {xv}')
        print(f'\ty_versor = {yv}')
        print(f'\tz_versor = {zv}')
        print('')

    def rot(self,angle:float=0,by:str='local z',deg=False):
        rot = RotMatrix(th=angle,axis=by[-1],deg=deg)
        R = TransformMatrix(rotation=rot)
        if by[0] == 'l': # local axis
            C = self * R
        elif by[0] == 'g': # global axis
            C = R * self
        else:
            raise ValueError('wrong by axis parameter')

        return C

    def trans(self,trans=[0,0,0],by:str='local x'):
        if type(trans) == float or type(trans) == int:
            d = {'x':0,'y':1,'z':2}
            t = [0,0,0]
            t[d[by[-1]]] = trans
            t = Vector(t[0],t[1],t[2])
        else:
            t = Vector(trans[0],trans[1],trans[2])
        R = TransformMatrix(translation=t)
        if by[0] == 'l': # local axis
            C = self * R
        elif by[0] == 'g': # global axis
            C = R * self
        else:
            raise ValueError('wrong by axis parameter')

        return C

    def get_rot(self):
        R = RotMatrix(R = np.matrix(self.C.A[:3,:3]))
        return R

    def get_versor(self,axis='z'):
        return self.get_rot().get_versor(axis=axis)

    def get_trans(self):
        return Vector(self.C[0,3],self.C[1,3],self.C[2,3])

    def plot(self,ax,no=None):
        '''
        lcs - local righthand coordinate system
        '''
        # get lcs vecor points
        center =   self.get_trans()
        C0 = self.get_versor(axis='x') + center
        C1 = self.get_versor(axis='y') + center
        C2 = self.get_versor(axis='z') + center
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
        if True:#param.lcs_coord:
            ax.text(center.x, center.y, center.z-0.1, (f'({center.x:.2f},{center.y:.2f},{center.z:.2f})'))

        return ax
        
    # def plot(self,ax,no=None,len=1):
    #     plot.lcs(ax,self,no,len)
    
@dataclass
class EulerAngle():

    def __init__(self,A :float = 0, B :float = 0, C :float = 0, deg=False, representation='zyz') -> None:
        factor = 1
        if deg:
            factor = FACTOR_DEG_2_RAD
        
        self.A = A * factor
        self.B = B * factor
        self.C = C * factor
        self.representation = representation

    def __str__(self) -> str:
        return f'({self.representation}) [A:{self.A:.4f}({self.A*FACTOR_RAD_2_DEG:.1f}[deg]), B:{self.B:.4f}({self.B*FACTOR_RAD_2_DEG:.1f}[deg]), C:{self.C:.4f}({self.C*FACTOR_RAD_2_DEG:.1f}[deg])]'

    def __repr__(self) -> str:
        return self.__str__()

    def to_rot_matrix(self):
        R0 = RotMatrix(th=self.A,axis=self.representation[0])
        R1 = RotMatrix(th=self.B,axis=self.representation[1])
        R2 = RotMatrix(th=self.C,axis=self.representation[2])
        R = R0*R1
        R *= R2
        return R

    def to_deg(self):
        return [self.A*FACTOR_RAD_2_DEG,self.B*FACTOR_RAD_2_DEG,self.C*FACTOR_RAD_2_DEG]

def intersection(p :Plane,s :Sphere):
    '''
    input:        plane = np.array([A,B,C,D])
                  sphere = np.array([x,y,z,r])
    output:       circle = np.array([xc,yc,zc,rc])
    '''
    
    # odległość pomiędzy środkiem sfery a płaszczyzną
    temp_1 = p.A*s.x+p.B*s.y+p.C*s.z+p.D
    temp_2 = p.A**2 + p.B**2 + p.C**2
    d = abs(temp_1) / np.sqrt(temp_2)

    if s.r >= d:
        # obliczanie współrzędnych okręgu
        xc = s.x - p.A*temp_1/temp_2
        yc = s.y - p.B*temp_1/temp_2
        zc = s.z - p.C*temp_1/temp_2
        rc = np.sqrt(s.r**2 - d**2); 
        circle = Circle(xc,yc,zc,rc)
        return circle
    else:
        raise Exception('the plane does not intersect the sphere')

def dist(p0 :Point, p1 :Point):
    return np.sqrt((p0.x-p1.x)**2 + (p0.y-p1.y)**2 + (p0.z-p1.z)**2)

def angle(A :Vector, B :Vector):
    A = A.to_array()
    B = B.to_array()
    v1 = np.array(A/np.linalg.norm(A))
    v2 = np.array(B/np.linalg.norm(B))
    return np.arccos(np.dot(v1,v2))

def deg2rad(joints):
    return [j*FACTOR_DEG_2_RAD for j in joints]

def rad2deg(joints):
    return [j*FACTOR_RAD_2_DEG for j in joints]