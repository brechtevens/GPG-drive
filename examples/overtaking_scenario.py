import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from GPGdrive.world import World
from GPGdrive.experiment import Experiment
from GPGdrive.visualize import Visualizer
import GPGdrive.dynamics as dynamics
import GPGdrive.settings as settings
import GPGdrive.collision as collision
import casadi as cs


def world_overtaking_scenario(solver_settings):
    """ Defines the set-up of an overtaking scenario

    Parameters
    ---------- 
    collision_mode : str, optional
        the constraint formulation, i.e. either 'projection', 'pointwise_projection', 'product' or 'dual'
    """
    # Initialize the world with a highway with 2 lanes
    world = World()
    world.set_nb_lanes(2)

    Ts=0.25
    N=12
    collision_mode='product'

    # Initialize the vehicles
    world.Ts = Ts
    dyn = dynamics.CarDynamics(Ts)

    id1 = world.add_vehicle('GPGOptimizerCar', dyn, [0., 0., 0., 6.5], N, gpg_solver_settings=solver_settings)
    id2 = world.add_vehicle('GPGOptimizerCar', dyn, [15, 0., 0., 5], N, gpg_solver_settings=solver_settings)

    # Select the rewards
    r1 = world.thesis_reward(1, 6.5, 0.01)[0]
    r2 = world.thesis_reward(1, 5., 1.)[0]

    # Set the rewards
    world.set_reward(id1, r1)
    world.set_reward(id2, r2)

    # Add the 'humans' to the corresponding vehicles
    world.add_human(id1, id2, r2)
    world.add_human(id2, id1, r1)

    # Add the common and the boundary constraints
    world.add_boundary_constraint(id1)
    world.add_boundary_constraint(id2)
    world.set_collision_avoidance_mode(collision_mode)

    return world

def experiment_overtaking_scenario():
    experiment = Experiment("overtaking_scenario")
    experiment.solver_settings.solver = 'panocpy'
    experiment.solver_settings.constraint_mode = 'hard'
    experiment.solver_settings.warm_start = True
    experiment.solver_settings.panoc_rebuild_solver = True
    experiment.solver_settings.panoc_use_alm = False

    experiment.world = world_overtaking_scenario(experiment.solver_settings)

    return experiment

if __name__ == '__main__':
    experiment = experiment_overtaking_scenario()
    vis = Visualizer(experiment)
    vis.run()