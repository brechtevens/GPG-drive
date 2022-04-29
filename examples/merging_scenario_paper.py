import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from GPGdrive.world import World
from GPGdrive.experiment import Experiment
from GPGdrive.visualize import Visualizer
import GPGdrive.dynamics as dynamics
import GPGdrive.settings as settings
import GPGdrive.collision as collision
import GPGdrive.feature as feature
import casadi as cs


def world_merging_scenario(solver_settings, learning_settings, is_human_courteous, is_belief_courteous):
    """ Defines the set-up of a merging scenario """
    Ts = 0.2
    N = 15
    collision_mode = 'product'

    # Initialize the world with a highway with 2 lanes 
    world = World()
    world.set_nb_lanes(2)

    # Initialize the vehicles
    world.Ts = Ts
    dyn_2d = dynamics.CarDynamics(world.Ts, lr=2, lf=2)
    dyn_1d = dynamics.CarDynamicsLongitudinal(world.Ts, lr=2, lf=2)

    id1 = world.add_vehicle('GPGOptimizerCar', dyn_2d, [3., 3., 0., 5.], N, gpg_solver_settings=solver_settings,
                            online_learning_settings=learning_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn_1d, [0., 0., 0., 5.], N, gpg_solver_settings=solver_settings)
    id3 = world.add_vehicle('UserControlledCar', dyn_1d, [7., 0., 0., 5.], N)
    world.cars[id3].fix_control([0.0])
    id4 = world.add_vehicle('UserControlledCar', dyn_1d, [45., 3., 0., 0], 1)
    world.cars[id4].fix_control([0.0])

    # Select the rewards
    p1 = cs.SX.sym('p1', 3, 1)
    p2 = cs.SX.sym('p2', 2, 1)
    pc = cs.SX.sym('pc', 3, 1)
    
    p1_num = [0.05, 0.1, 0.5]
    pc_num = [4., 2., 1.5]
    p2_control = 0.1

    r_shared = pc[0] * feature.gaussian_aligned(id1, id2, height=pc[1], width=pc[2])
    r_p1 = pc[0] * feature.gaussian_aligned(id1, id3, height=pc[1], width=pc[2]) - p1[0] * world.scene.quadratic() + feature.control_individual(p1[1], p1[2])
    r_p2 = p2[0] * world.cars[id2].traj.quadratic_following_reward(3., world.cars[id3]) + p2[1] * feature.control()

    # Set the rewards
    world.set_reward(id1, r_p1, params=cs.vertcat(p1, pc), param_values=[*p1_num, *pc_num], shared_reward=r_shared)
    if is_human_courteous:
        p2_num = [0.02, p2_control] 
    else:
        p2_num = [10, p2_control]

    world.set_reward(id2, r_p2, params=cs.vertcat(p2, pc), param_values=[*p2_num, *pc_num], shared_reward=r_shared)

    # Add the 'humans' and the 'obstacles' to the corresponding vehicles
    if is_belief_courteous:
        p2_num_belief = [0.02, p2_control]
    else:
        p2_num_belief = [10, p2_control]
    
    world.add_human(id1, id2, r_p2, params=p2, param_values=p2_num_belief)

    world.add_obstacle(id1, id3)
    world.add_obstacle(id1, id4)
    world.add_human(id2, id1, r_p1, params=p1, param_values=p1_num)
    world.add_obstacle(id2, id3)
    world.add_obstacle(id2, id4)

    # Add the common and the boundary constraints
    world.add_boundary_constraint(id1)
    world.set_collision_avoidance_mode(collision_mode)
    return world


def experiment_merging_scenario(is_learning, is_human_courteous, is_belief_courteous):
    experiment = Experiment("merging_scenario", [str(is_learning), str(is_human_courteous), str(is_belief_courteous)])

    ## Logger settings
    experiment.logger_settings.nb_iterations_experiment = 55

    ## Visualisation settings
    experiment.pyglet_visualization_settings.magnify = 1.0
    experiment.pyglet_visualization_settings.camera_offset = [20, 1.5]
    experiment.pyglet_visualization_settings.width_factor = 2100/1000
    experiment.pyglet_visualization_settings.height_ratio = 7
    experiment.pyglet_visualization_settings.id_main_car = 2

    ## Solver settings
    experiment.solver_settings.solver = 'panocpy'
    experiment.solver_settings.constraint_mode = 'hard'
    experiment.solver_settings.warm_start = True
    experiment.solver_settings.panoc_rebuild_solver = True
    experiment.solver_settings.max_time = 0.1

    # PANOC settings
    experiment.solver_settings.panoc_initial_penalty = 0.1
    experiment.solver_settings.panoc_delta_tolerance = 0.01
    experiment.solver_settings.panoc_tolerance = 1e-3

    ## Learning settings
    if is_learning:
        experiment.learning_settings.solver = 'panocpy'
        experiment.learning_settings.nb_observations = 1
        experiment.learning_settings.regularization = 0.5
        experiment.learning_settings.panoc_rebuild_solver = True
        experiment.learning_settings.based_on_original_gpg = False
        experiment.learning_settings.max_time = 0.1

        experiment.learning_settings.panoc_tolerance = 1  # OpEn default = 1e-5
        experiment.learning_settings.panoc_initial_tolerance = 1
        experiment.learning_settings.panoc_delta_tolerance_primal_feas = 1e-3
        experiment.learning_settings.panoc_delta_tolerance_complementarity = 1e-3
        experiment.learning_settings.panoc_delta_tolerance_complementarity_bounds = 1e-3

    ## Build world
    experiment.world = world_merging_scenario(experiment.solver_settings, experiment.learning_settings, is_human_courteous, is_belief_courteous)
    experiment.data_visualization_windows = settings.default_data_visualization_windows(experiment.world)
    return experiment

if __name__ == '__main__':
    experiment = experiment_merging_scenario(True, True, True)
    vis = Visualizer(experiment)
    vis.run()
