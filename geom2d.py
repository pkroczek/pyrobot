from dataclasses import dataclass
import numpy as np

@dataclass
class Point():

    x :float = 0
    y :float = 0

    def __str__(self) -> str:
        return f'point [x:{self.x :.4f},y:{self.y :.4f}]'

    def __repr__(self) -> str:
        return self.__str__()

    def to_array(self):
        return np.array([self.x,self.y])

    def norm(self):
        return np.sqrt(self.x**2 + self.y**2)

@dataclass
class Line():

    a :float = 0
    b :float = 0
    p0 : Point = Point()

    def __str__(self) -> str:
        return f'line [a:{self.a :.4f},b:{self.b :.4f},p0: [{self.p0}]]'

    def __repr__(self) -> str:
        return self.__str__()

    def from_points(self, p0 :Point, p1: Point):
        self.a = p1.x - p0.x
        self.b = p1.y - p0.y
        self.p0 = p0

    def get_point_at(self, t :float=0):
        return Point(self.p0.x + self.a*t,self.p0.y + self.b*t)

@dataclass
class Circle():

    x :float = 0
    y :float = 0
    r :float = 0

    def __str__(self) -> str:
        return f'sphere [x:{self.x :.4f},y:{self.y :.4f},r:{self.r :.4f}]'

    def __repr__(self) -> str:
        return self.__str__()

    def to_array(self):
        return np.array([self.x,self.y,self.z,self.r])

    def to_point(self):
        return Point(self.x,self.y)

def dist(p0 :Point, p1 :Point):
    return np.sqrt((p0.x-p1.x)**2 + (p0.y-p1.y)**2)

def intersection(c0 :Circle, c1 :Circle):
    '''
    input:        c1, c2 = np.array([xc,yc,r])
    output:       p[0] = np.array([x1,y1])
                  p[1] = np.array([x2,y2])
    source: http://www.ambrsoft.com/TrigoCalc/Circles2/circle2intersection/CircleCircleIntersection.htm
    '''

    # obliczenie odległości pomiędzy środkami okręgów
    D = dist(c0.to_point(),c1.to_point())

    if ( (c0.r+c1.r) >= D) and (D >= abs(c0.r-c1.r)): # warunek przecinania się okręgów

        area = .25*np.sqrt((D+c0.r+c1.r)*(D+c0.r-c1.r)*(D-c0.r+c1.r)*(-D+c0.r+c1.r))

        x = (c0.x+c1.x)/2 + ((c1.x-c0.x)*(c0.r**2-c1.r**2))/(2*D**2) + 2*(c0.y-c1.y)/D**2*area
        y = (c0.y+c1.y)/2 + ((c1.y-c0.y)*(c0.r**2-c1.r**2))/(2*D**2) - 2*(c0.x-c1.x)/D**2*area
        p0 = Point(x,y)

        x = (c0.x+c1.x)/2 + ((c1.x-c0.x)*(c0.r**2-c1.r**2))/(2*D**2) - 2*(c0.y-c1.y)/D**2*area
        y = (c0.y+c1.y)/2 + ((c1.y-c0.y)*(c0.r**2-c1.r**2))/(2*D**2) + 2*(c0.x-c1.x)/D**2*area
        p1 = Point(x,y)

        return p0,p1

    else:
        raise ValueError(f'circles do not intersect')

def intersection(c :Circle, l :Line):
    '''
    Circle form: (x-xc)^2 + (y-yc)^2 = r^2
    Line form: y = ax + b
    --------------------------------------------------
    input:
    c = [xc,yc,r],  (xc,yc) - center of circle, 
                          r - radius of circle
    l = [a, b],      a, b - line coefficient
    --------------------------------------------------
    output:
    res = [[x0,y0],[x1,y1]]
    --------------------------------------------------
    source: http://www.ambrsoft.com/TrigoCalc/Circles2/circlrLine_.htm
    '''
    delta = c.r**2*(1+l.a**2) - (c.y - l.a*c.x - l.b)**2
    x0 = (c.x + c.y*l.a - l.b*l.a + np.sqrt(delta))/(1 + l.a**2)
    x1 = (c.x + c.y*l.a - l.b*l.a - np.sqrt(delta))/(1 + l.a**2)
    y0 = (l.b + c.x*l.a + c.y*l.a**2 + l.a*np.sqrt(delta))/(1 + l.a**2)
    y1 = (l.b + c.x*l.a + c.y*l.a**2 - l.a*np.sqrt(delta))/(1 + l.a**2)
    p0 = Point(x0,y0)
    p1 = Point(x1,y1)

    return p0,p1

