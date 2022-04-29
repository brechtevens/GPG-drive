from . import car as car

class PygletVisualizationSettings(object):
    """
        A class used to represent the logger settings

        Attributes
        ----------
        magnify : float
            the magnification factor
        height_ratio : float
            the ratio width/height of the pyglet window
        width_factor : float
            the magnification factor of the width, factor 1 equals 1000 pixels
        camera_offset : list
            offset of the camera center
    """
    def __init__(self):
        self.magnify = 0.5
        self.height_ratio = 4
        self.width_factor = 1
        self.camera_offset = [10., 0]
        self.show_live_data = None
        self.live_data_box_position = (10, 10)
        self.live_data_border_size = 10
        self.id_main_car = 0


def default_data_visualization_windows(world):
    id_list = [vehicle.id for vehicle in world.cars]
    id_list_gpg = [vehicle.id for vehicle in world.cars if isinstance(vehicle, car.GPGOptimizerCar)]
    default = {
        'velocity': id_list,
        'acceleration': id_list,
        'steering angle': id_list,
        'cost': id_list_gpg,
        'potential': [],
        'effective constraint violation': id_list_gpg
        #'belief': [0, 1]
    }
    return default


class LoggerSettings(object):
    """
        A class used to represent the logger settings

        Attributes
        ----------
        nb_iterations_experiment : int
            the number of iterations of the experiment to log
        name_experiment : str
            the name of the experiment
        save_video : boolean
            determines whether a video of the current experiment should be saved
        only_save_statistics : boolean
            determines whether only the computation statistics of the current experiment should be saved
        statistics_index : int
            the index of the resulting statistics file
    """
    def __init__(self, name_experiment):
        self.nb_iterations_experiment = 100
        self.name_experiment = name_experiment
        self.save_video = True
        self.only_save_statistics = False
        self.statistics_index = None


class SolverSettings(object):
    """
        A class used to represent solver settings

        Attributes
        ----------
        solver : str
            the used solver for the optimal control problems of the GNEPOptimizerCars, i.e. 'ipopt', 'OpEn' or 'panocpy'
        ipopt_tolerance : float
            the tolerance for the ipopt solver
        ipopt_acceptable_tolerance : float
            the acceptable tolerance for the ipopt solver
        panoc_rebuild_solver : boolean
            determines whether the solver needs to be recompiled, set False for successive experiments with same code
        open_use_python_bindings : boolean
            whether to use python bindings or tcp sockets for the OpEn solver
        panoc_build_mode : str
            build mode for OpEn and panocpy: 'debug' = fast compilation, 'release' = fast runtime
        panoc_tolerance : float
            the tolerance of the applied solver
        panoc_initial_tolerance : float
            the initial tolerance of the applied solver
        panoc_delta_tolerance : float
            the delta tolerance of the applied solver, i.e. ||F2(x)||_inf < delta
        panoc_initial_penalty : int
            the initial value of the penalty parameters
        panoc_penalty_update_factor : int
            the update factor alpha for the penalty parameters
        panoc_max_outer_iterations : int
            the maximum amount of outer ALM/quadratic penalty iterations
        panoc_max_inner_iterations : int
            the maximum amount of inner PANOC iterations
        open_directory_name : str
            name of the subdirectory in og_builds for storing the compiled code
    """
    solver = 'panocpy'
    max_time = 5                            # OpEn default 5

    ipopt_tolerance = 1e-6                  # ipopt default 1e-8
    ipopt_acceptable_tolerance = 1e-4

    open_use_python_bindings = True

    panoc_rebuild_solver = True
    panoc_build_mode = 'debug'
    panoc_tolerance = 1e-4                  # OpEn default = 1e-4
    panoc_initial_tolerance = 1             # Tolerance on fixed point residual, inf-norm
    panoc_delta_tolerance = 1e-4            # OpEn default 1e-4, ||F2(x)||_inf < delta
    panoc_initial_penalty = 1               # OpEn default 1
    panoc_penalty_weight_update_factor = 5  # OpEn default 5
    panoc_max_outer_iterations = 25         # OpEn default 10
    panoc_max_inner_iterations = 500        # OpEn default 500
    
    panoc_lbfgsmem = 20
    panoc_use_alm = False

    def __init__(self, directory_name="default"):
        self.open_directory_name = directory_name

class GPGSolverSettings(SolverSettings):
    """
        A class used to represent the gpg solver settings

        Attributes
        ----------
        shooting_mode : str
            the shooting mode for the optimal control problems of the GNEPOptimizerCars, i.e. 'single' or 'multiple'
        use_gauss_seidel : boolean
            whether to use Gauss-Seidel or the direct solver
        constraint_mode : str
            the constraint mode for the common constraints of the GPG formulations, i.e. 'hard' or 'soft'
        penalty_method : str
            which penalty method to use, i.e. 'quadratic' or 'exact'
        penalty_update_rule : str
            which update rule to use, i.e. 'individual' or 'collective'
        warm_start : boolean
            determines whether penalty parameters are warm started
        gs_max_iterations : int
            the maximum number of iterations for the Gauss-Seidel best-response algorithm
        gs_regularization : float
            the regularization parameter tau for the Gauss-Seidel best-response algorithm
        gs_tolerance : float
            the stopping criterion for the Gauss-Seidel outer iterations, delta_solution < game_tolerance
    """
    shooting_mode = 'single'
    use_gauss_seidel = False

    gs_max_iterations = 50
    gs_tolerance = 1e-2
    gs_regularization = 0.

    constraint_mode = 'hard'
    penalty_method = 'quadratic'
    penalty_update_rule = 'individual'

    warm_start = False
        
class OnlineLearningSettings(SolverSettings):
    """
        A class used to represent the online learning settings

        Attributes
        ----------
        regularization : float
            the regularization parameter of the online learning methodology
        nb_observations : int
            the amount of previous observations used to update the estimate
        based_on_original_gpg : boolean
            whether to base the optimization problem on the real on the penalized GPG
        panoc_delta_tolerance_primal_feas : float
            delta tolerance for the primal feasability
        panoc_delta_tolerance_complementarity : float
            delta tolerance for the complementarity constraints
        panoc_delta_tolerance_complementarity_bounds : float
            delta tolerance for the complementarity constraints of the bounds
    """
    regularization = 1
    nb_observations = 0
    based_on_original_gpg = False

    max_time = 5                            # OpEn default 5

    ipopt_tolerance = 1e-3                  # ipopt default 1e-8
    ipopt_acceptable_tolerance = 1e-1

    panoc_tolerance = 1e-6                  # OpEn default = 1e-4
    panoc_initial_tolerance = 1e-4          # Tolerance on fixed point residual, inf-norm
    panoc_delta_tolerance = 1e-2
    panoc_initial_penalty = 1
    panoc_penalty_weight_update_factor = 2
    panoc_max_outer_iterations = 20
    panoc_max_inner_iterations = int(1e5)
    panoc_use_alm = True

    panoc_delta_tolerance_primal_feas = 1e-2
    panoc_delta_tolerance_complementarity = 1e-3
    panoc_delta_tolerance_complementarity_bounds = 1e-3



        
        

