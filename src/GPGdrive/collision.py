import math
import casadi as cs
from . import constraints as constraints
from . import boundingbox as boundingbox

from . import car as car
from . import dynamics as dynamics
import numpy as np
import matplotlib.pyplot as plt
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def terminal_constraint(car1, car2, A, b):
    """ Returns the terminal constraints A * [car1.x, car2.x] >= b

    Used to implement the RCI set in the one-dimensional GPG

    Parameters
    ----------
    car1 : Car object
        the first car for the constraint formulation
    car2 : Car object
        the second car for the constraint formulation
    A : list of lists
        matrix A
    b : list
        vector b
    """
    # Return constraints Ax - b >= 0
    @constraints.stageconstraints
    def h(x, u=None):
        constraint_list = []
        for i in range(len(A)):
            constraint_list.append(A[i][0] * x[car1.id][0] + A[i][1] * x[car1.id][3] +
                                   A[i][2] * x[car2.id][0] + A[i][3] * x[car2.id][3] - b[i])
        return constraint_list
    h.length = len(A)
    h.id_list = tuple(sorted((car1.id, car2.id)))
    return h


def bounds_constraint(car, lbx, ubx):
    """ Manually implements the bounds constraint for vehicles with longitudinal dynamics

    Parameters
    ----------
    lbx : float
        lower bound on the acceleration
    ubx : float
        upper bound on the acceleration
    """
    @constraints.stageconstraints
    def h(x, u=None):
        constraint = [(ubx - u[car.id]) * (u[car.id] + lbx)]
        return constraint
    h.length = 1
    h.id_list = (car.id,)
    return h


def headway_formulation_constraint(car, target, d_des = 0):
    """ Returns a constraint enforcing that the minimal headway distance is respected between two vehicles

    This implementation is only valid for a 1D road section in the x directioon

    Parameters
    ----------
    car : Car object
        the ego vehicle
    target : Car object
        the other vehicle
    d_des : float, optional
        the desired minimal headway distance
    """
    if car.x[0] > target.x[0]:
        front = 'self'
        lr_front = car.lr
        lf_back = target.lf
    else:
        front = 'target'
        lr_front = target.lr
        lf_back = car.lf

    @constraints.stageconstraints
    def h(x, u=None):
        if front == 'self':
            constraint = [x[car.id][0] - lr_front - x[target.id][0] - lf_back - d_des]
        else:
            constraint = [x[target.id][0] - lr_front - x[car.id][0] - lf_back - d_des]
        return constraint
    h.length = 1
    h.id_list = tuple(sorted((car.id, target.id)))
    return h


def projection_formulation_squared_distance():
    """ Returns a function for the squared distance """
    def squared_distance(p_r, p_f, y, smooth=False):
        """ Returns the squared distance between a point p and an aligned bounding box

        Parameters
        ----------
        p_r : list
            the lower bounding point of the aligned bounding box
        p_f : list
            the upper bounding point of the aligned bounding box
        y : list or cs.DM
            the position of a point p
        """
        if smooth:
            def sigmoid(z):
                return 1/(1 + cs.exp(z))
            def smooth_max(x, y, k=10):
                sig = sigmoid(-k*(x - y))
                return x * sig + y * (1 - sig)
            def smooth_min(x,y):
                return - smooth_max(-x, -y)
            return cs.sumsqr(y - smooth_max(smooth_min(y,  p_f), p_r))
        else:
            return cs.sumsqr(y - cs.fmax(cs.fmin(y,  p_f), p_r))
    return squared_distance


def projection_formulation_inequality_constraints(car1, car2, epsilon=0.1, smooth=False):
    """ Returns the inequality constraints for the projection formulation

    Parameters
    ----------
    car1 : Car object
        the first vehicle
    car2 : Car object
        the second vehicle
    epsilon : float, optional
        the minimal distance required between both vehicles, i.e. virtual enlargement
    """
    squared_dist = projection_formulation_squared_distance()

    @constraints.stageconstraints
    def h(x, u=None):
        p_r, p_f = boundingbox.get_axis_aligned_bounding_box(car1, car2, x[car1.id], x[car2.id])
        return [squared_dist(p_r, p_f, x[car1.id][0:2], smooth) - epsilon]
    h.length = 1
    h.id_list = tuple(sorted((car1.id, car2.id)))
    return h

def ellipse_formulation_inequality_constraints(car1, car2, epsilon=0.1, smooth=False):
    """ Returns the inequality constraints for the ellipse formulation

    Parameters
    ----------
    car1 : Car object
        the first vehicle
    car2 : Car object
        the second vehicle
    epsilon : float, optional
        the minimal distance required between both vehicles, i.e. virtual enlargement
    """

    @constraints.stageconstraints
    def h(x, u=None):
        E = boundingbox.get_axis_alligned_ellipsoid(car1, car2, x[car1.id], x[car2.id])
        center1 = car1.center_x(x[car1.id])
        center2 = car2.center_x(x[car2.id])
        return [(center1[0:2] - center2[0:2]).T @ E @ (center1[0:2] - center2[0:2]) - 1]
    h.length = 1
    h.id_list = tuple(sorted((car1.id, car2.id)))
    return h

# pointwise projection formulation aligned
def pointwise_projection_formulation_aligned_squared_distance():
    """ Returns a function for the squared distance """
    def squared_distance(target_vehicle, x_target, y):
        """ Returns the squared distance between a point p and a vehicle, described by a rectangle which is aligned

        Parameters
        ----------
        target_vehicle : Car object
            the target vehicle
        x_target : cs.DM
            the position of the target vehicle
        y : list or cs.DM
            the position of a point p
        """
        x_c_target = target_vehicle.center_x(x_target)[0:2]
        if isinstance(y, list):
            y = cs.DM(y)
        symbol_initializer = getattr(cs, y.type_name())
        x_max = symbol_initializer([(target_vehicle.dyn.lr + target_vehicle.dyn.lf)/2., target_vehicle.dyn.width/2.])
        return cs.sumsqr((y - x_c_target) - cs.fmin(cs.fmax((y - x_c_target),  -x_max), x_max))
    return squared_distance

# pointwise projection formulation
def pointwise_projection_formulation_squared_distance():
    """ Returns a function for the squared distance """
    def squared_distance(target_vehicle, x_target, y):
        """ Returns the squared distance between a point p and a vehicle, described by a rectangle

        Parameters
        ----------
        target_vehicle : Car object
            the target vehicle
        x_target : cs.DM
            the position of the target vehicle
        y : list or cs.DM
            the position of a point p
        """
        x_c_target = target_vehicle.center_x(x_target)[0:2]
        if isinstance(y, list):
            y = cs.DM(y)
        symbol_initializer = getattr(cs, y.type_name())
        x_max = symbol_initializer([(target_vehicle.dyn.lr + target_vehicle.dyn.lf)/2., target_vehicle.dyn.width/2.])
        R_transpose = cs.vertcat(cs.horzcat(cs.cos(x_target[2]), cs.sin(x_target[2])),
                                 cs.horzcat(-cs.sin(x_target[2]), cs.cos(x_target[2])))
        return cs.sumsqr(R_transpose @ (y - x_c_target) -
                         cs.fmin(cs.fmax(R_transpose @ (y - x_c_target),  -x_max), x_max))
    return squared_distance


def pointwise_projection_formulation_inequality_constraints(car1, car2, epsilon=1e-2):
    """ Returns the 8 inequality constraints for the pointwise projection formulation

    Parameters
    ----------
    car1 : Car object
        the first vehicle
    car2 : Car object
        the second vehicle
    epsilon : float, optional
        the minimal distance required between both vehicles, i.e. virtual enlargement
    """
    squared_dist = pointwise_projection_formulation_squared_distance()

    @constraints.stageconstraints
    def h(x, u=None):
        constraint_list = []
        for corner in car1.corners_x(x[car1.id]):
            constraint_list.append(squared_dist(car2,x[car2.id],corner) - epsilon**2)
        for corner in car2.corners_x(x[car2.id]):
            constraint_list.append(squared_dist(car1,x[car1.id],corner) - epsilon**2)
        return constraint_list
    h.length = 8
    h.id_list = tuple(sorted((car1.id, car2.id)))
    return h


def affine_edges(target_vehicle, x_target):
    """ Returns the A and b matrices defining the affine edges of the target vehicle

    Parameters
    ----------
    target_vehicle : Car object
        the target vehicle
    x_target : cs.DM
        the position of the target vehicle
    """
    target_corners = target_vehicle.corners_x(x_target)
    homogeneous_target_corners = [cs.vertcat([1], corner) for corner in target_corners]
    edges = cs.horzcat(cs.cross(homogeneous_target_corners[0], homogeneous_target_corners[1], 1)/target_vehicle.width,
                       cs.cross(homogeneous_target_corners[1], homogeneous_target_corners[2], 1)/target_vehicle.len,
                       cs.cross(homogeneous_target_corners[2], homogeneous_target_corners[3], 1)/target_vehicle.width,
                       cs.cross(homogeneous_target_corners[3], homogeneous_target_corners[0], 1)/target_vehicle.len)
    A = cs.transpose(edges[1:3,:])
    B = -cs.transpose(edges[0, :])
    return A, B


def product_formulation_affine_constraints_of_target_vehicle():
    """ Returns the 8 equality constraints for the product formulation """
    def affine_constraints(target_vehicle, x_target, y):
        A, B = affine_edges(target_vehicle, x_target)
        distances = B - A @ y
        return distances
    return affine_constraints


def product_formulation_equality_constraint_of_target_vehicle(epsilon):
    """ Returns the function for the equality constraints of the product formulation

    Parameters
    ----------
    epsilon : float, optional
        the minimal distance required between both vehicles, i.e. virtual enlargement
    """
    affine_constraints = product_formulation_affine_constraints_of_target_vehicle()

    def equality_constraint(target_vehicle, x_target, y):
        """ Returns the value of the equality constraint of the product formulation

        Parameters
        ----------
        target_vehicle : Car object
            the target vehicle
        x_target : cs.DM
            the position of the target vehicle
        y : list or cs.DM
            the position of a point p
        """
        eval_affine = affine_constraints(target_vehicle, x_target, y)
        return - cs.fmax(eval_affine[0] + epsilon, 0) * cs.fmax(eval_affine[1] + epsilon, 0) * \
               cs.fmax(eval_affine[2] + epsilon, 0) * cs.fmax(eval_affine[3] + epsilon, 0)
    return equality_constraint

# def product_formulation_equality_constraints(car1, car2, epsilon=1e-1):
#     """ Returns the 8 equality constraints for the product formulation

#     Parameters
#     ----------
#     car1 : Car object
#         the first vehicle
#     car2 : Car object
#         the second vehicle
#     epsilon : float, optional
#         the minimal distance required between both vehicles, i.e. virtual enlargement
#     """
#     g_single_point = product_formulation_equality_constraint_of_target_vehicle(epsilon)

#     @constraints.stageconstraints
#     def g(x, u=None):
#         constraint_list = []
#         for corner in car1.corners_x(x[car1.id]):
#             constraint_list.append(g_single_point(car2, x[car2.id], corner))
#         for corner in car2.corners_x(x[car2.id]):
#             constraint_list.append(g_single_point(car1, x[car1.id], corner))
#         # return [cs.mmin(np.array(constraints))]
#         return constraint_list
#     g.length = 8
#     g.id_list = tuple(sorted((car1.id, car2.id)))
#     return g

def product_formulation_equality_constraints(car1, car2, epsilon=1e-4):
    """ Returns the 10 equality constraints for the product formulation

    Parameters
    ----------
    car1 : Car object
        the first vehicle
    car2 : Car object
        the second vehicle
    epsilon : float, optional
        the minimal distance required between both vehicles, i.e. virtual enlargement
    """
    g_single_point = product_formulation_equality_constraint_of_target_vehicle(epsilon)

    @constraints.stageconstraints
    def g(x, u=None):
        constraint_list = []
        for corner in car1.corners_x(x[car1.id]):
            constraint_list.append(g_single_point(car2, x[car2.id], corner))
        constraint_list.append(10*g_single_point(car2, x[car2.id], car1.front_x(x[car1.id])))
        for corner in car2.corners_x(x[car2.id]):
            constraint_list.append(g_single_point(car1, x[car1.id], corner))
        constraint_list.append(10*g_single_point(car1, x[car1.id], car2.front_x(x[car2.id])))
        return constraint_list
    g.length = 10
    g.id_list = tuple(sorted((car1.id, car2.id)))
    return g

def product_formulation_equality_constraints_simplified(car1, car2, epsilon=1e-1):
    """ Returns 1 equality constraints for the product formulation

    Parameters
    ----------
    car1 : Car object
        the first vehicle
    car2 : Car object
        the second vehicle
    epsilon : float, optional
        the minimal distance required between both vehicles, i.e. virtual enlargement
    """
    g_single_point = product_formulation_equality_constraint_of_target_vehicle(epsilon)

    @constraints.stageconstraints
    def g(x, u=None):
        constraint_list = [g_single_point(car2, x[car2.id], x[car1.id][0:2])]
        return constraint_list
    g.length = 1
    g.id_list = tuple(sorted((car1.id, car2.id)))
    return g

def dual_formulation_constraints(car1, car2, epsilon=1e-2):
    """ Returns the 10 constraints for the dual formulation

    Parameters
    ----------
    car1 : Car object
        the first vehicle
    car2 : Car object
        the second vehicle
    epsilon : float, optional
        the minimal distance required between both vehicles, i.e. virtual enlargement
    """
    @constraints.stageconstraints
    def g_and_h(x, lam, mu, u=None):
        A1, B1 = affine_edges(car1, x[car1.id])
        A2, B2 = affine_edges(car2, x[car2.id])
        h = [-cs.dot(B1,lam) - cs.dot(B2, mu) - epsilon,
             1-cs.norm_2(cs.transpose(A1) @ lam)**2,
             lam[0], lam[1], lam[2], lam[3],
             mu[0], mu[1], mu[2], mu[3]]
        g = [cs.dot(A1[:, 0], lam) + cs.dot(A2[:, 0], mu),
             cs.dot(A1[:, 1], lam) + cs.dot(A2[:, 1], mu)]
        return g, h
    g_and_h.length = 10
    g_and_h.id_list = tuple(sorted((car1.id, car2.id)))
    return g_and_h


if __name__ == '__main__':
    dyn = dynamics.CarDynamics(0.1)
    car1 = car.GPGOptimizerCar(dyn, [0., 0., math.pi / 2., 5.], 20, 0, 'red')
    car2 = car.GPGOptimizerCar(dyn, [1., 3., math.pi / 2., 5.], 20, 1, 'yellow')
    angle = math.pi/2

    # Tests for squared distance function
    def assertSXEquals(SXvalue, value, precision):
        assert (value - 10 ** (-precision) <= SXvalue <= value + 10 ** (-precision))
    collision_function = pointwise_projection_formulation_squared_distance()
    assertSXEquals(collision_function(car1, car1.x, [0, 0]), 0, 7)
    assertSXEquals(collision_function(car1, car1.x, [-1, 0]), 0, 7)
    assertSXEquals(collision_function(car1, car1.x, [1, 0]), 0, 7)
    assertSXEquals(collision_function(car1, car1.x, [-1, -4]), 0, 7)
    assertSXEquals(collision_function(car1, car1.x, [1, -4]), 0, 7)
    assertSXEquals(collision_function(car1, car1.x, [-1, 1]), 1, 7)
    assertSXEquals(collision_function(car1, car1.x, [2, 1]), 2, 7)
    assertSXEquals(collision_function(car1, car1.x, [0, 2]), 4, 7)
    assertSXEquals(collision_function(car1, car1.x, [3, -2]), 4, 7)

    # General function to test formulations
    def test_formulation(name_formulation, formulation, angle, mode):
        plt.figure()
        plt.title('Test ' + name_formulation + ', width = 2, length = 4')
        x_grid = np.linspace(-6, 6, 10)
        y_grid = np.linspace(-6, 6, 10)
        results = np.zeros([len(x_grid), len(y_grid)])
        for x_i in range(len(x_grid)):
            for y_i in range(len(y_grid)):
                if mode == 'equality':
                    results[x_i][y_i] = all(i == 0 for i in formulation({0: [0., 0., math.pi / 2., 5.], 1: [x_grid[x_i], y_grid[y_i], angle, 5.]}))
                elif mode == 'inequality':
                    results[x_i][y_i] = all(i >= 0 for i in formulation({0: [0., 0., math.pi / 2., 5.], 1: [x_grid[x_i], y_grid[y_i], angle, 5.]}))
                else:
                    results[x_i][y_i] = formulation([0., 0., math.pi / 2., 5.], [x_grid[x_i], y_grid[y_i], angle, 5.])
                if results[x_i][y_i]:
                    plt.scatter(x_grid[x_i], y_grid[y_i], color='green')
                else:
                    plt.scatter(x_grid[x_i], y_grid[y_i], color='red')
        plt.show()

    # tests for pointwise_projection formulation
    pointwise_projection_formulation_inequality_constraints = pointwise_projection_formulation_inequality_constraints(car1, car2)
    test_formulation('pointwise_projection formulation', pointwise_projection_formulation_inequality_constraints, angle, 'inequality')

    # Tests for product formulation
    product_formulation_equality_constraints = product_formulation_equality_constraints(car1, car2)
    test_formulation('product formulation', product_formulation_equality_constraints, angle, 'inequality')

    # # Tests for dual formulation
    # # Setup variables
    # x_r = cs.SX.sym('x_r', dyn.nx)
    # x_h = cs.SX.sym('x_h', dyn.nx)
    # dual_constraints = dual_formulation_constraints(car1,car2)
    # lam = cs.SX.sym('lambda', 4, 1)
    # mu = cs.SX.sym('mu', 4, 1)
    # params = cs.vertcat(lam, mu)
    # x_dict = {car1.id: x_r, car2.id: x_h}
    # g,h = dual_constraints(x_dict, lam, mu)
    # collision_checker = {'x': params, 'f': cs.norm_2(params)**2, 'p': cs.vertcat(x_r, x_h), 'g': cs.vertcat(np.array(g), np.array(h))}
    # collision_checker_solver = cs.nlpsol('Collision_checker', 'ipopt', collision_checker, {'verbose_init': False, 'print_time': False, 'ipopt': {'print_level': 0}})
    # nb_g = len(g)
    # nb_h = len(h)
    # ubg = [0] * nb_g + [float("inf")] * nb_h
    # def dual_formulation_check_solution(pos_vehicle1, pos_vehicle2):
    #     solution = collision_checker_solver(x0=np.array([0] * 8),
    #                                         p=cs.vertcat(pos_vehicle1, pos_vehicle2), lbg=0,
    #                                         ubg=ubg)
    #     solution['g'] = np.round(solution['g'], 7)
    #     return all(i == 0 for i in solution['g'][1:nb_g]) and all(i >= 0 for i in solution['g'][nb_g:])
    #
    # test_formulation('dual formulation', dual_formulation_check_solution, angle, 'dual')

    test_car = car.GPGOptimizerCar(dyn, [2., 0., 0., 5.], 'red', id=0, T=20)
    squared_distance = pointwise_projection_formulation_squared_distance()
    x = np.arange(-3.0, 3.01, 0.025)
    y = np.arange(-2.0, 2.01, 0.025)
    X, Y = np.meshgrid(x, y)  # grid of point
    Z = np.zeros([len(X), len(X[0])])
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z[i, j] = squared_distance(test_car, test_car.x, [X[i, j], Y[i, j]])

    fig = plt.figure()
    im = imshow(Z, cmap=cm.Reds, extent=[-3, 3, -2, 2])  # drawing the function
    # adding the Contour lines with labels
    cset = contour(Z, np.arange(0, 2, 0.5), extent=[-3, 3, -2, 2], linewidths=2, colors='black')
    clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    colorbar(im, fraction=0.0315, pad=0.02, format='%1.1f', ticks=[0, 0.5, 1, 1.5, 2])  # adding the colobar on the right
    # latex fashion title
    # title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
    plt.yticks([-2, -1, 0, 1, 2])
    plt.ylabel('y [m]', fontsize=12)
    plt.xlabel('x [m]', fontsize=12)
    show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.RdBu, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    h = product_formulation_equality_constraint_of_target_vehicle(0)
    Z_product = np.zeros([len(X), len(X[0])])
    for i in range(len(X)):
        for j in range(len(X[0])):
            Z_product[i, j] = - h(test_car, test_car.x, [X[i, j], Y[i, j]])

    fig = plt.figure()
    im = imshow(Z_product, cmap=cm.Reds, extent=[-3, 3, -2, 2])  # drawing the function
    # adding the Contour lines with labels
    cset = contour(Z_product, np.arange(0, 4.01, 1), extent=[-3, 3, -2, 2], linewidths=2, colors='black')
    clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    colorbar(im, fraction=0.0315, pad=0.02, format='%1.1f', ticks=[0, 1, 2, 3, 4])  # adding the colobar on the right
    # latex fashion title
    # title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
    plt.yticks([-2, -1, 0, 1, 2])
    plt.ylabel('y [m]', fontsize=12)
    plt.xlabel('x [m]', fontsize=12)
    show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z_product, rstride=1, cstride=1,
                           cmap=cm.RdBu, linewidth=0, antialiased=False)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
