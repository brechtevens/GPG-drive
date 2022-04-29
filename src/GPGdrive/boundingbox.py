import casadi as cs
import warnings


def get_axis_aligned_deltas(car, x):
        
    def b(W_L, k):
        return (W_L/k) * (1 - cs.tanh(W_L))

    def phi(x, k):
        def W_L_1():
            # return special.lambertw(1)
            return 0.5671432904097838
        W_L = W_L_1()
        return b(W_L, k) + x * cs.tanh(k * x)

    x = car.x if x is None else x

    cos_psi = cs.cos(x[2])
    sin_psi = cs.sin(x[2])
    abs_cos_psi = phi(cos_psi, 20)
    abs_sin_psi = phi(sin_psi, 20)
    delta_xf = car.lf * abs_cos_psi + (car.width/2)*abs_sin_psi
    delta_xr = car.lr * abs_cos_psi + (car.width/2)*abs_sin_psi
    # delta_yf = car.lf * abs_sin_psi + (car.width/2)*cos_psi
    # delta_yr = car.lr * abs_sin_psi + (car.width/2)*cos_psi
    delta_yf = (car.lf - car.len/2) * sin_psi + (car.len/2) * abs_sin_psi + (car.width/2)*abs_cos_psi
    delta_yr = (car.lr - car.len/2) * sin_psi + (car.len/2) * abs_sin_psi + (car.width/2)*abs_cos_psi
    return delta_xf, delta_xr, delta_yf, delta_yr

def get_axis_aligned_box(car, x=None):
    x = car.x if x is None else x
    delta_xf, delta_xr, delta_yf, delta_yr = get_axis_aligned_deltas(car, x)

    p_f = cs.vertcat(x[0] + delta_xf, x[1] + delta_yf)
    p_r = cs.vertcat(x[0] - delta_xr, x[1] - delta_yr)
    return p_r, p_f

def get_axis_aligned_bounding_box(car1, car2, x1=None, x2=None):
    x1 = car1.x if x1 is None else x1
    x2 = car2.x if x2 is None else x2   

    delta_xf1, delta_xr1, delta_yf1, delta_yr1 = get_axis_aligned_deltas(car1, x1)
    delta_xf2, delta_xr2, delta_yf2, delta_yr2 = get_axis_aligned_deltas(car2, x2)

    p_f = cs.vertcat(x2[0] + delta_xr1 + delta_xf2, x2[1] + delta_yr1 + delta_yf2)
    p_r = cs.vertcat(x2[0] - delta_xf1 - delta_xr2, x2[1] - delta_yf1 - delta_yr2)
    return p_r, p_f

def get_axis_alligned_ellipsoid(car1, car2, x1=None, x2=None):
    x1 = car1.x if x1 is None else x1
    x2 = car2.x if x2 is None else x2  

    p_r, p_f = get_axis_aligned_bounding_box(car1, car2, x1, x2)
    gamma = 7
    delta_x = ((p_f[0] - p_r[0])/2)
    delta_y = ((p_f[1] - p_r[1])/2)
    lambda_1 = 1/(delta_x**2 + gamma*delta_y**2)

    E = cs.diag(cs.vertcat(lambda_1, gamma*lambda_1))
    return E

def get_axis_alligned_ellipsoid_vertices(car1, car2, x1=None, x2=None):
    x1 = car1.x if x1 is None else x1
    x2 = car2.x if x2 is None else x2  

    p_r, p_f = get_axis_aligned_bounding_box(car1, car2, x1, x2)
    gamma = 7
    center_x = ((p_f[0] + p_r[0])/2)
    center_y = ((p_f[1] + p_r[1])/2)
    delta_x = ((p_f[0] - p_r[0])/2)
    delta_y = ((p_f[1] - p_r[1])/2)
    lambda_1 = 1/(delta_x**2 + gamma*delta_y**2)
    
    return center_x, center_y, lambda_1, gamma*lambda_1

def get_axis_aligned_box_vertices(car):
    p_r, p_f = get_axis_aligned_box(car)
    return [[p_f[0], p_f[1]], [p_f[0], p_r[1]], [p_r[0], p_r[1]], [p_r[0], p_f[1]]]

def get_axis_aligned_bounding_box_vertices(car1, car2):
    p_r, p_f = get_axis_aligned_bounding_box(car1, car2)
    return [[p_f[0], p_f[1]], [p_f[0], p_r[1]], [p_r[0], p_r[1]], [p_r[0], p_f[1]]]