import numpy as np
import sympy as sym
from pyrobot.geom3d import RotMatrix, Vector, TransformMatrix, FACTOR_RAD_2_DEG
import pyrobot.plot as plot
import pyrobot.geom3d_sym as g3s

def dh(rotZ:float=0,transZ:float=0,transX:float=0,rotX:float=0,param=None):
    '''
    Obliczenie numeryczne macierzy transformacji Denavita-Hartenberga
    param = [rotZ transZ transX rotX]
    '''
    if param is not None and len(param) >= 4:
        rotZ = param[0]
        transZ = param[1]
        transX = param[2]
        rotX = param[3]
    Rz = TransformMatrix(Vector(),RotMatrix(rotZ,axis='z'))
    Tz = TransformMatrix(Vector(0,0,transZ),RotMatrix(axis='none'))
    Tx = TransformMatrix(Vector(transX,0,0),RotMatrix(axis='none'))
    Rx = TransformMatrix(Vector(),RotMatrix(rotX,axis='x'))

    R = Rz*Tz*Tx*Rx
    return R

def dh_sym(rotZ:sym.Symbol=0,transZ:sym.Symbol=0,transX:sym.Symbol=0,rotX:sym.Symbol=0,param=None):
    '''
    Obliczenie numeryczne macierzy transformacji Denavita-Hartenberga
    param = [rotZ transZ transX rotX]
    '''
    if param is not None and len(param) >= 4:
        rotZ = param[0]
        transZ = param[1]
        transX = param[2]
        rotX = param[3]
    Rz = g3s.TransformMatrix(g3s.Vector(),g3s.RotMatrix(rotZ,axis='z'))
    Tz = g3s.TransformMatrix(g3s.Vector(0,0,transZ),g3s.RotMatrix(axis='none'))
    Tx = g3s.TransformMatrix(g3s.Vector(transX,0,0),g3s.RotMatrix(axis='none'))
    Rx = g3s.TransformMatrix(g3s.Vector(),g3s.RotMatrix(rotX,axis='x'))

    R = Rz*Tz*Tx*Rx
    return R    

def normalize_angle(angle :float):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

class Manipulator():

    def __init__(self,dh_table_fcn,inv_kin_fcn,param,plot_table=None) -> None:
        self.dh_table_fcn = dh_table_fcn
        self.plot_table = plot_table
        self.inv_kin_fcn = inv_kin_fcn
        self.param = param

    def fwd(self,joints) -> TransformMatrix:
        if type(joints[0]) is sym.Symbol:
            _dh = dh_sym
            C = g3s.TransformMatrix()
        else:
            _dh = dh
            C = TransformMatrix()
        
        table = self.dh_table_fcn(joints=joints,param=self.param)
        for t in table:
            C *= _dh(param=t)

        if type(joints[0]) is sym.Symbol:
            C = C.simplify()
        return C

    def inv(self, R :TransformMatrix):
        return self.inv_kin_fcn(R,self.param)

    def resutls(self, R :TransformMatrix=None, joints=None) -> str:
        if R is not None:
            joints,_ = self.inv(R)
        elif joints is None:
            raise ValueError('wrong input data type')

        joints_desc = ''
        for i,j in enumerate(joints):
            joints_desc += f'DOF{i:d}: {j:.2f}({j*FACTOR_RAD_2_DEG:.2f}[deg]),'
        
        target = self.fwd(joints=joints)
        orientation_desc = str(target.get_rot().to_zyz())
        position_desc = str(target.get_trans())

        res = f'JOINTS:\t\t{joints_desc}\nORIENTATION:\t{orientation_desc}\nPOSITION:\t{position_desc}'
        return res

    def plot(self, ax, R :TransformMatrix=None, joints=None, length=10):
        if R is not None:
            joints,_ = self.inv(R)
        elif joints is None:
            raise ValueError('wrong input data type')
        
        plot.title(self.resutls(R,joints))
        table = self.dh_table_fcn(joints=joints,param=self.param)
        
        if self.plot_table is None:
            self.plot_table = ['k'] * len(table)

        centers = []
        C = TransformMatrix()
        plot.lcs(ax,C,0,length)
        centers.append(C.get_trans())

        for i,t in enumerate(table):
            C *= dh(param=t)
            centers.append(C.get_trans())
            if i < len(table)-1:
                coord=False
            else:
                coord=True

            if self.plot_table[i] is not None and len(centers) >= 2:
                plot.lcs(ax,C,i+1,length,coord)
                plot.line(ax,centers[-1],centers[-2],color=self.plot_table[i],linewidth=3)


if __name__ == "__main__":                
    J1, J2, J3, J4, ARM1, ARM2, GEAR = sym.symbols('J1 J2 J3 J4 ARM1 ARM2 GEAR')
    a = dh_sym(J1,J2,J3,J4)
    print(a)