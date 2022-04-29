import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from GPGdrive.world import World
from GPGdrive.experiment import Experiment
from GPGdrive.visualize import Visualizer
import GPGdrive.dynamics as dynamics
import GPGdrive.settings as settings
import GPGdrive.collision as collision
import casadi as cs
import math

def world_one_dimensional_gpg(solver_settings, learning_settings, is_belief_correct):
    """ Defines the set-up of a one-dimensional GPG with parameter estimation """
    Ts = 0.25
    N = 12

    # Initialize the world with a highway with a single lane
    world = World()
    world.set_nb_lanes(1)

    # Initialize the cost function parameters
    d_min = 0
    a_min = 3.0
    a_max = 3.0
    a_min_p = cs.SX.sym('a_min', 1, 1)
    a_max_p = cs.SX.sym('a_max', 1, 1)

    # Initialize the vehicles
    world.Ts = Ts
    dyn1 = dynamics.CarDynamicsLongitudinal(Ts)
    dyn2 = dynamics.CarDynamicsLongitudinal(Ts, bounds=[[-a_min_p], [a_max_p]])

    id1 = world.add_vehicle('GPGOptimizerCar', dyn1, [0., 0., 0., 5], N,
                            gpg_solver_settings=solver_settings, online_learning_settings=learning_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn2, [19., 0., 0., 5], N,
                            gpg_solver_settings=solver_settings)

    # Select the rewards
    r1, r_p1, p1 = world.thesis_reward(0.1, 7, dynamics=dyn1)
    r2, r_p2, p2 = world.thesis_reward(0.1, 5, dynamics=dyn2)

    # Set the rewards
    world.set_reward(id1, r1)
    world.set_reward(id2, r2, params=cs.vertcat(a_min_p, a_max_p), param_values=[a_min, a_max])

    # Add the 'humans' to the corresponding vehicles
    if is_belief_correct:
        world.add_human(id1, id2, r_p2, params=cs.vertcat(p2, a_min_p, a_max_p), param_values=[0.1, 5., a_min, a_max])
    else:
        world.add_human(id1, id2, r_p2, params=cs.vertcat(p2, a_min_p, a_max_p), param_values=[0.2, 7., 0.1, 0.1])
    world.add_human(id2, id1, r1)

    # Add the common constraints and the bounds for the human
    world.add_common_constraints(collision.headway_formulation_constraint, 'add_h')
    # world.set_collision_avoidance_mode('ellipse')
    return world

def experiment_one_dimensional_gpg(is_learning, is_belief_correct):
    experiment = Experiment("one_dimensional_gpg")

    ## Solver settings
    experiment.solver_settings.solver = 'panocpy'
    experiment.solver_settings.constraint_mode = 'hard'
    experiment.solver_settings.panoc_rebuild_solver = True
    experiment.solver_settings.open_use_python_bindings = True
    experiment.solver_settings.use_gauss_seidel = False

    # Gauss-Seidel legacy settings
    experiment.solver_settings.panoc_initial_penalty = 10 #exact 0.1, quadratic 10
    experiment.solver_settings.panoc_penalty_weight_update_factor = 2 #exact and quadratic 2
    experiment.solver_settings.gs_tolerance = 1e-3
    experiment.solver_settings.penalty_method = 'quadratic'
    experiment.solver_settings.panoc_max_outer_iterations = 20
    experiment.solver_settings.gs_max_iterations = 100

    # OpEn settings
    experiment.solver_settings.panoc_tolerance = 1e-5
    experiment.solver_settings.panoc_delta_tolerance = 1e-5

    ## Learning settings
    if is_learning:
        experiment.learning_settings.solver = 'panocpy'
        experiment.learning_settings.nb_observations = 1
        experiment.learning_settings.regularization = [1e2, 1e0, 1e5, 1e5]
        experiment.learning_settings.panoc_rebuild_solver = True
        experiment.learning_settings.based_on_original_gpg = True
        # experiment.learning_settings.panoc_tolerance = 1e-6
        # experiment.learning_settings.panoc_initial_tolerance = 1e-5
        # experiment.learning_settings.panoc_delta_tolerance = 1e-5
        # experiment.learning_settings.panoc_delta_tolerance_primal_feas = 1e-5
        # experiment.learning_settings.panoc_delta_tolerance_complementarity = 1e-6
        # experiment.learning_settings.panoc_delta_tolerance_complementarity_bounds = 1e-7
        experiment.learning_settings.panoc_tolerance = 1e-4
        experiment.learning_settings.panoc_initial_tolerance = 1e-3
        experiment.learning_settings.panoc_delta_tolerance = 1e-3
        experiment.learning_settings.panoc_delta_tolerance_primal_feas = 1e-3
        experiment.learning_settings.panoc_delta_tolerance_complementarity = 1e-4
        experiment.learning_settings.panoc_delta_tolerance_complementarity_bounds = 1e-5

    # Build world
    experiment.world = world_one_dimensional_gpg(experiment.solver_settings, experiment.learning_settings, is_belief_correct)
    experiment.data_visualization_windows = settings.default_data_visualization_windows(experiment.world)
    return experiment

if __name__ == '__main__':
    experiment = experiment_one_dimensional_gpg(True, False)
    vis = Visualizer(experiment)
    vis.run()