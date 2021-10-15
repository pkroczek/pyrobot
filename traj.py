from dataclasses import dataclass
import numpy as np
# from pyrobot.geom3d import RotMatrix, Vector, TransformMatrix, EulerAngle
from geom3d import RotMatrix, Vector, TransformMatrix, EulerAngle
import geom3d


@dataclass
class MoveParam():

    a :float=1
    v :float=1
    d :float=1

    def __str__(self) -> str:
        return f'[Vel:{self.v:.2f}, Acc:{self.a:.2f}, Dec:{self.d:.2f}]'

    @classmethod
    def from_times(self, s, t, ta, td):
        if ta + td >= t:
            v = 2*s/(ta+td)
        else:
            # z pola powierzchni trapezu (a+b)/2 * h
            # total_distance = (total_move_time + (total_move_time - acc_time - dec_time))/2 * vel
            v = 2*s / (2*t - ta - td)

        a = v/ta
        d = v/td
        return MoveParam(a,v,d)

    def get_time(self,s :float=1):
        # s - distance to move
        # v - velocity
        # a - acceleration
        # d - deceleration
        
        # v = at
        # s = s0 + v0*t + (a*a)*t/2

        # obliczenie czasu potrzebnego na przyspieszanie
        ta = self.v/self.a
        sa = self.a*ta*ta/2
        # obliczenie czasu potrzebnego na wyhamowanie
        td = self.v/self.d
        sd = self.d*td*td/2

        # pozostała droga do przebycia w ruchu jednostajnym
        sv = s - (sa + sd)

        # jeżeli droga pokonana w trakcie przyspieszania i hamowania
        # jest mniejsza niż połowa całkowitej drogi
        if sv > 0:

            tv = sv/self.v
            t = ta+tv+td

        # jeżeli droga do przebycia nie pozwala na osiągnięcie 
        # docelowej prędkości
        else:
            # 1: sa/sd = d/a
            # 1: sa = (d/a)*sd
            # 2: s = sa + sd
            # 3: z 1 i 2: s = (d/a + 1)*sd
            # 3: sd = s / (d/a + 1)

            sd = s / (self.d/self.a + 1)
            sa = s - sd

            # 4: sa = (a*t*t)/2
            ta = np.sqrt(2*sa/self.a)
            td = np.sqrt(2*sd/self.d)

            # max osiągnięta prędkość
            v = self.a * ta
            # całkowity czas przejazdu
            t = ta+td
            # przebyta trasa dla weryfikacji 
            s = t*v/2 

            tv = 0

        return t, ta, tv, td

    def plot(self,ax,s :float=1):
        t, ta, tv, td = self.get_time(s)
        ax.plot([0,ta,t-td,t],[0,self.v,self.v,0])
        ax.set_xlabel('time [s]')
        ax.set_ylabel('velocity [m/s]')

    def generate_time_vector(self,s,n:int=10):
        t, _, _, _ = self.get_time(s)
        return np.linspace(0,t,n)

    def position_at_time(self,s:float=1,n:int=10,initial_pos:float=0):

        # obliczenie czasu potrzebnego na przyspieszanie
        ta = self.v/self.a
        sa = self.a*ta*ta/2
        # obliczenie czasu potrzebnego na wyhamowanie
        td = self.v/self.d
        sd = self.d*td*td/2
        # pozostała droga do przebycia w ruchu jednostajnym
        sv = s - (sa + sd)
        # jeżeli droga pokonana w trakcie przyspieszania i hamowania
        # jest mniejsza niż połowa całkowitej drogi
        if sv > 0:
            tv = sv/self.v
        # jeżeli droga do przebycia nie pozwala na osiągnięcie 
        # docelowej prędkości
        else:
            # 1: sa/sd = d/a
            # 1: sa = (d/a)*sd
            # 2: s = sa + sd
            # 3: z 1 i 2: s = (d/a + 1)*sd
            # 3: sd = s / (d/a + 1)

            sd = s / (self.d/self.a + 1)
            sa = s - sd

            # 4: sa = (a*t*t)/2
            ta = np.sqrt(2*sa/self.a)
            td = np.sqrt(2*sd/self.d)

            # max osiągnięta prędkość
            v = self.a * ta
            # całkowity czas przejazdu
            t = ta+td
            # czas w ruchu jednostajnym
            tv = 0

        time_vector = self.generate_time_vector(s,n)
        pos = [0]*n
        for i,t in enumerate(time_vector):
            # etap rozpędzania
            if t <= ta:
                #
                p = self.a*t*t/2
            
            # etap stałej prędkości
            elif t <= ta+tv:
                # odjęcie czasu na przyspieszanie
                t -= ta
                # suma drogi przebytej w trakcie przyspieszania + pozostały czas razy prędkość
                p = sa + self.v*t

            # etap hamowania
            else:
                # odjęcie czasu przyspieszania i ruchu z jednostajną prędkością
                t -= (ta+tv)
                # bieżąca prędkość
                v_ = self.v - self.d*t
                if v_ > 0:
                    # suma drogi przebytej w trakcie przyspieszania i ze stałą prędkością
                    p = sa + sv
                    # przebyta droga w trakcie hamowania
                    p += (self.v*t) - (self.d*t*t/2)
                else:
                    p = s

            pos[i] = p+initial_pos

        dt = time_vector[1] - time_vector[0]
        return dt, time_vector, pos

class Trajectory():

    def __init__(self,P :Vector=Vector(), R :EulerAngle=EulerAngle()) -> None:
        self.start_point = [P,R]
        self.stop_points = []
        self.traj = []

    def add(self,P :Vector=Vector(), R :EulerAngle=EulerAngle(), M :MoveParam=MoveParam(), n :int=10):
        self.stop_points.append([P,R,M,n])

    def _gen_one_axis(self,A,Z,t,ta,td,n):
        dist = abs(A-Z)
        mp = MoveParam.from_times(dist,t,ta,td)
        dt,_,p = mp.position_at_time(s=dist,n=n,initial_pos=A)
        return dt,p
        

    def _gen_all_axis(self,start,stop,t,ta,td,n:int):
        dt = [[]]*6
        dt[0],x = self._gen_one_axis(start[0].x,stop[0].x,t,ta,td,n)
        dt[1],y = self._gen_one_axis(start[0].y,stop[0].y,t,ta,td,n)
        dt[2],z = self._gen_one_axis(start[0].z,stop[0].z,t,ta,td,n)
        dt[3],a = self._gen_one_axis(start[1].A,stop[1].A,t,ta,td,n)
        dt[4],b = self._gen_one_axis(start[1].B,stop[1].B,t,ta,td,n)
        dt[5],c = self._gen_one_axis(start[1].C,stop[1].C,t,ta,td,n)
        for i in dt:
            if i is not np.NAN:
                dt = i
                break
        res = np.array([x,y,z,a,b,c]).transpose()
        return res,dt

    def generate(self):
        self.traj = []
        current_t = 0
        P0 = self.start_point
        for sp in self.stop_points:
            # przypisanie punktu docelowego
            P1 = sp
            # obliczenie odległości do pokonania w przestrzeni kartezjańskiej
            dist3d = geom3d.dist(P0[0],P1[0])
            # wyznaczenie czasu ruchu 
            t, ta, tv, td = sp[2].get_time(dist3d)
            # wygenerowanie poszczególnych wektorów zmiany parametrów
            table, dt = self._gen_all_axis(P0,P1,t,ta,td,n=sp[3])

            for i,t in enumerate(table):
                R = TransformMatrix(Vector(t[0],t[1],t[2]),EulerAngle(t[3],t[4],t[5]).to_rot_matrix())
                self.traj.append([current_t,R])
                current_t += dt
            P0 = P1

    @property
    def size(self):
        N = 0
        for s in self.stop_points:
            N += s[3]

        return N

    @property
    def time(self):
        T = 0
        P0 = self.start_point[0]
        for sp in self.stop_points:
            P1 = sp[0]
            s = geom3d.dist(P0,P1)
            t = sp[2].get_time(s)
            T += t[0]
            P0 = P1
        return T

if __name__ == "__main__":

    traj = Trajectory()
    traj.add(n=10,P=Vector(10,10,1))        
    traj.add(n=70,P=Vector(90,-50,1))
    traj.generate()
    print(traj.size)        
    print(traj.time)        

