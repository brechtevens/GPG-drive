import casadi as cs
from . import collision as collision

class Feature(object):
    """
    A class used to represent cost function features
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self.f(*args)

    def __add__(self, r):
        return Feature(lambda *args: self(*args)+r(*args))

    def __radd__(self, r):
        return Feature(lambda *args: r(*args)+self(*args))

    def __mul__(self, r):
        return Feature(lambda *args: self(*args)*r)

    def __rmul__(self, r):
        return Feature(lambda *args: r*self(*args))

    def __pos__(self, r):
        return self

    def __neg__(self):
        return Feature(lambda *args: -self(*args))

    def __sub__(self, r):
        return Feature(lambda *args: self(*args)-r(*args))

    def __rsub__(self, r):
        return Feature(lambda *args: r(*args)-self(*args))


def feature(f):
    """ Decorator function """
    return Feature(f)

def empty():
    """ Returns a quadratic reward feature penalizing deviations from the desired speed """
    @feature
    def f(x, u, x_other, k):
        return 0
    return f

def speed(s=1.):
    """ Returns a quadratic reward feature penalizing deviations from the desired speed """
    @feature
    def f(x, u, x_other, k):
        return -(x[3]-s)*(x[3]-s)
    return f

def reference(reference_x, reference_y):
    """ Returns a quadratic reward feature penalizing deviations from the desired reference trajectory """
    @feature
    def f(x, u, x_other, k):
        return - (x[0]-reference_x[k])*(x[0]-reference_x[k]) - (x[1]-reference_y[k])*(x[1]-reference_y[k])
    return f

def destination(destination):
    """ Returns a quadratic reward feature penalizing deviations from the desired target destination """
    @feature
    def f(x, u, x_other, k):
        return - (x[0]-destination[0])*(x[0]-destination[0]) - (x[1]-destination[1])*(x[1]-destination[1])
    return f

def control(parametric_bounds=None):
    """ Returns a quadratic reward feature penalizing control actions """
    if parametric_bounds is None:
        @feature
        def f(x, u, x_other, k):
            return -cs.sumsqr(u)
    else:
        @feature
        def f(x, v, x_other, k):
            u = parametric_bounds[0][0] + (1+v[0])/2*(parametric_bounds[1][0] - parametric_bounds[0][0])    
            return -cs.sumsqr(u)
    return f

def control_individual(p_accel, p_angle):
    @feature
    def f(x, u, x_other, k):
        return - p_accel * u[0]**2 - p_angle * u[1]**2
    return f

def headway(front, lr_front, lf_back):
    """ Returns a reward feature penalizing driving to close in a one-dimensional scenario

    Parameters
    ----------
    front : boolean
        indicates whether the ego vehicle is the front or the rear vehicle
    lr_front : float
        the distance between the center of mass and the rear end of the front vehicle
    lf_back : float
        the distance between the center of mass and the front end of the rear vehicle
    """
    @feature
    def f(x, u, x_other, k):
        if front:
            d_x = x[0]-x_other[0]-lr_front-lf_back
        else:
            d_x = x_other[0]-x[0]-lr_front-lf_back
        return d_x
    return f


def gaussian(id_other, height=4., width=2.):
    """ Returns a gaussian reward feature rewarding distance between the ego and the target vehicle

    Parameters
    ----------
    id_other : int
        the id of the target vehicle
    height : float
        the size of the gaussian in the longitudinal direction
    width : float
        the size of the gaussian in the lateral direction
    """
    @feature
    def f(x, u, x_other, k):
        d = (x_other[id_other][0]-x[0], x_other[id_other][1]-x[1])
        theta = x_other[id_other][2]
        dh = cs.cos(theta)*d[0]+cs.sin(theta)*d[1]
        dw = -cs.sin(theta)*d[0]+cs.cos(theta)*d[1]
        return -cs.exp(-0.5*(dh*dh/(height*height)+dw*dw/(width*width)))
    return f

def gaussian_aligned(id_1, id_2, height=3., width=2):
    """ Returns a gaussian reward feature rewarding distance between the ego and the target vehicle

    Parameters
    ----------
    id_other : int
        the id of the target vehicle
    height : float
        the size of the gaussian in the longitudinal direction
    width : float
        the size of the gaussian in the lateral direction
    """
    @feature
    def f(x, u, x_dict, k):
        d = (x_dict[id_1][0]-x_dict[id_2][0], x_dict[id_1][1]-x_dict[id_2][1])
        dh = d[0]
        dw = d[1]
        return -cs.exp(-0.5*(dh*dh/(height*height)+dw*dw/(width*width)))
    return f

def smooth_exponential(x):
    return cs.if_else(x>0,cs.exp(-1/x),0)

def transition_function(x):
    return smooth_exponential(x) / (smooth_exponential(x) + smooth_exponential(1-x))

def smoothstep(x):
    return cs.if_else(x>=1,1,cs.if_else(x<=0,0,3*x**2-2*x**3))

def rectangular_full(car_other, id_other, dist):
    pointwise_projection = collision.pointwise_projection_formulation_squared_distance()
    @feature
    def f(x, u, x_other, k):
        dist_squared = pointwise_projection(car_other, x_other[id_other], x[0:2])
        # return transition_function(dist_squared/dist**2)
        return smoothstep(dist_squared/dist**2)
    return f

def rectangular_aligned(car_other, id_other, dist):
    pointwise_projection = collision.pointwise_projection_formulation_aligned_squared_distance()
    @feature
    def f(x, u, x_other, k):
        dist_squared = pointwise_projection(car_other, x_other[id_other], x[0:2])
        return transition_function(dist_squared/dist**2)
    return f

