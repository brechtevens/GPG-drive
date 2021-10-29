import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataloader import loadMatrix
import scipy.io

from src.world import World
from src.experiment import Experiment
import src.dynamics as dynamics
import src.settings as settings
import src.collision as collision
from src.visualize import Visualizer
import casadi as cs
import types


def world_racing(solver_settings, learning_settings, is_belief_correct):
    """ Defines the set-up of a one-dimensional GPG with parameter estimation """
    Ts = 0.01
    N = 20

    # Initialize the world with a highway with a single lane
    world = World()
    world.set_racetrack()

    # Initialize the vehicles
    world.Ts = Ts
    dyn = dynamics.CarDynamics(Ts, bounds = [[-3., -2], [3., 2]], lr=0.13, lf=0.0, width=0.065)

    id1 = world.add_vehicle('GPGOptimizerCar', dyn, [0., 0.1, 0., 1.5], N,
                            gpg_solver_settings=solver_settings, online_learning_settings=learning_settings)
    # id2 = world.add_vehicle('GPGOptimizerCar', dyn, [0., -0.1, 0., 1.5], N,
    #                         gpg_solver_settings=solver_settings)

    # Select the rewards
    ref_x = cs.SX.sym('ref_x', N, 1)
    ref_y = cs.SX.sym('ref_y', N, 1)
    _, r_p1, p1 = world.scene.tracking_reward(1, ref_x, ref_y)
    # _, r_p2, p2 = world.scene.tracking_reward(1, ref_x, ref_y)

    # Set the rewards
    world.set_reward(id1, r_p1, params = cs.vertcat(p1, ref_x, ref_y), param_values = [100] + world.scene.get_reference(N, 0))
    # world.set_reward(id2, r_p2, params = cs.vertcat(p2, ref_x, ref_y), param_values = [100] + world.scene.get_reference(N, 0))
 
    def update(self, k):
        self.ego.reward_params_current_belief = [100] + world.scene.get_reference(N, k)

    world.cars[id1].update = types.MethodType(update, world.cars[id1])
    # world.cars[id2].update = types.MethodType(update, world.cars[id2])


    # Add the 'humans' to the corresponding vehicles
    # world.add_human(id1, id2, r_p2, params=p2, param_values=[100])
    # world.add_human(id2, id1, r_p1, params=p1, param_values=[100])

    # Add the common constraints and the bounds for the human
    # world.set_collision_avoidance_mode('product')
    return world

def experiment_racing(is_learning, is_belief_correct):
    experiment = Experiment('racetrack')

    experiment.logger_settings.save_video = False
    experiment.logger_settings.nb_iterations_experiment = 400

    experiment.pyglet_visualization_settings.id_main_car = None
    experiment.pyglet_visualization_settings.magnify = 0.1
    experiment.pyglet_visualization_settings.camera_offset = [0.25, 0.75]

    ## Solver settings
    experiment.solver_settings.solver = 'OpEn'
    experiment.solver_settings.constraint_mode = 'hard'
    experiment.solver_settings.panoc_rebuild_solver = True
    experiment.solver_settings.open_use_python_bindings = True
    experiment.solver_settings.use_gauss_seidel = False

    # OpEn settings
    experiment.solver_settings.panoc_tolerance = 1e-4
    experiment.solver_settings.panoc_delta_tolerance = 1e-5
    experiment.solver_settings.panoc_initial_penalty = 1e6

    ## Learning settings
    if is_learning:
        experiment.learning_settings.solver = 'panocpy'
        experiment.learning_settings.nb_observations = 4
        experiment.learning_settings.regularization = [1e2, 1e0, 1e4, 1e4]
        experiment.learning_settings.panoc_rebuild_solver = True
        experiment.learning_settings.based_on_original_gpg = True
        experiment.learning_settings.panoc_tolerance = 1e-4
        experiment.learning_settings.panoc_initial_tolerance = 1e-3
        experiment.learning_settings.panoc_delta_tolerance = 1e-3
        experiment.learning_settings.panoc_delta_tolerance_primal_feas = 1e-3
        experiment.learning_settings.panoc_delta_tolerance_complementarity = 1e-4
        experiment.learning_settings.panoc_delta_tolerance_complementarity_bounds = 1e-5

    # Build world
    experiment.world = world_racing(experiment.solver_settings, experiment.learning_settings, is_belief_correct)
    experiment.data_visualization_windows = settings.default_data_visualization_windows(experiment.world)
    return experiment
5
if __name__ == '__main__':
    experiment = experiment_racing(False, True)
    vis = Visualizer(experiment)
    vis.run()



# if __name__ == '__main__':
#     centerX, centerY, theta = load_reference_track('c')

#     import pyglet
#     from pyglet.window import key
#     import pyglet.gl as gl
#     import pyglet.graphics as graphics

#     height_ratio = 2
#     window = pyglet.window.Window(1000, int(1000/height_ratio), fullscreen=False)
#     magnify = 1
    
#     # override the method that draws when the window loads
#     @window.event
#     def on_draw():
#         window.clear()
#         gl.glColor3f(1., 1., 1.)
#         gl.glMatrixMode(gl.GL_PROJECTION)
#         gl.glPushMatrix()
#         gl.glLoadIdentity()
#         gl.glOrtho(*[-2/magnify,3/magnify,-1/height_ratio/magnify,4/height_ratio/magnify], 1., -1.)
#         draw_line(centerX, centerY)
#         print('draw')

#     def on_mouse_scroll(x, y, scroll_x, scroll_y):
#         magnify *= (1 - 0.1*scroll_y)
#         print('scroll')

#     def draw_line(xlist, ylist):
#         # gl.glColor3f(0,0,0)
#         gl.glBegin(gl.GL_LINES)
#         for point in zip(xlist, ylist):
#             gl.glVertex2f(*point)
#             # print(point)
#         gl.glEnd()
#         # gl.glColor3f(1,1,1)

#         # gl.glBegin(gl.GL_LINE_LOOP)
#         # # create a line, x,y,z
#         # gl.glVertex2f(100.0,100.0)
#         # gl.glVertex2f(200.0,300.0)
#         # gl.glVertex2f(*point)
#         # gl.glEnd()

#     window.on_mouse_scroll = on_mouse_scroll

#     pyglet.app.run()

