from dataclasses import dataclass
import numpy as np
import sympy as sym
from sympy import pi
from sympy.matrices import *

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sym.Number)})

@dataclass
class Point():

    x :sym.Symbol=0
    y :sym.Symbol=0
    z :sym.Symbol=0

    def __str__(self) -> str:
        return f'[x:{self.x}, y:{self.y}, z:{self.z}]'

    def __repr__(self) -> str:
        return self.__str__()

    def to_array(self):
        return np.array([self.x,self.y,self.z])

    def norm(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

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

    R :Matrix

    def __init__(self,th:sym.Symbol=0,axis :str='none',R:Matrix=None) -> None:

        if R is not None and R.shape == (3,3):
            self.R = R
        else:
            rot = {'x':self.rotX,'y':self.rotY,'z':self.rotZ,'none':self.eye}
            self.R = rot[axis](th)

    def __str__(self) -> str:
        t  =   f'[{self.R[0,0]}, {self.R[0,1]}, {self.R[0,2]}]'
        t += f'\n[{self.R[1,0]}, {self.R[1,1]}, {self.R[1,2]}]'
        t += f'\n[{self.R[2,0]}, {self.R[2,1]}, {self.R[2,2]}]'
        return t

    def __mul__(self,other):
        R =  self.R * other.R
        return RotMatrix(R=R)

    def get_versor(self,axis='z'):
        d = {'x':0,'y':1,'z':2}
        V = self.R[:,d[axis]]
        return Vector(V[0],V[1],V[2])

    def rotX(self,th):
        return sym.transpose(rot_axis1(th))

    def rotY(self,th):
        return sym.transpose(rot_axis2(th))

    def rotZ(self,th):
        return sym.transpose(rot_axis3(th))

    def eye(self,th):
        return eye(3)

@dataclass
class TransformMatrix():

    C :Matrix

    def __init__(self,translation :Vector=Vector(), rotation :RotMatrix=RotMatrix(), C :np.matrix=None) -> None:

        if C is not None and C.shape == (4,4):
            self.C = C
        else:
            C = rotation.R
            C = C.row_join(Matrix(3,1,[translation.x,translation.y,translation.z]))
            C = C.col_join(Matrix(1,4,[0,0,0,1]))
            self.C = C

    def __str__(self) -> str:
        t  =   f'[{self.C[0,0]}, {self.C[0,1]}, {self.C[0,2]}, {self.C[0,3]}]'
        t += f'\n[{self.C[1,0]}, {self.C[1,1]}, {self.C[1,2]}, {self.C[1,3]}]'
        t += f'\n[{self.C[2,0]}, {self.C[2,1]}, {self.C[2,2]}, {self.C[2,3]}]'
        t += f'\n[{self.C[3,0]}, {self.C[3,1]}, {self.C[3,2]}, {self.C[3,3]}]'
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


    def simplify(self):
        self.C = round_expr(self.C,4)
        return TransformMatrix(C=self.C.simplify())

    def rot(self,angle:sym.Symbol,by:str='local z'):
        rot = RotMatrix(th=angle,axis=by[-1])
        R = TransformMatrix(rotation=rot)
        if by[0] == 'l': # local axis
            C = self * R
        elif by[0] == 'g': # global axis
            C = R * self
        else:
            raise ValueError('wrong by axis parameter')

        return C

    def trans(self,trans=[0,0,0],by:str='local x'):
        if type(trans) == sym.Symbol:
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
        R = RotMatrix(R = Matrix(self.C[:3,:3]))
        return R

    def get_versor(self,axis='z'):
        return self.get_rot().get_versor(axis=axis)

    def get_trans(self):
        return Vector(self.C[0,3],self.C[1,3],self.C[2,3])
