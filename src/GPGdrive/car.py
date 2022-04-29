import numpy as np
from . import gaussseidelsolver as gaussseidelsolver
from . import learning as learning
from . import lagrangiansolver as lagrangiansolver
from .penalty import PenaltyParameters
from .trajectory import Trajectory
import casadi as cs
from collections import deque
import time
from . import settings as settings
from .player import Obstacle, Player
from . import feature as feature

class Car(object):
    """
    A class used to represent general car objects

    Attributes
    ----------
    id : int
        the id of the vehicle
    reset_x0 : list
        the initial state of the vehicle
    N : int
        the control horizon of the vehicle
    dyn : Dynamics object
        the dynamics of the vehicle
    traj : Trajectory object
        the trajectory object of the vehicle
    color : str
        the color of the vehicle
    default_u : list
        the default control sequence applied by the vehicle

    Properties
    ----------
    x : list
        the current state of the vehicle
    center_x : list
        the current position of the center of the vehicle
    u : list
        the current control action of the vehicle
    corners : list
        list of the current position of the four corners of the vehicle
    lr : float
        the length between the mass center and the rear end
    lf : float
        the length between the mass center and the front end
    len : float
        the length of the vehicle
    width : float
        the width of the vehicle
    u_2D : list
        the current control action of the vehicle, given by [acceleration, steering angle]
    """
    def __init__(self, car_dyn, x0, N, id, color):
        """
        Parameters
        ----------
        car_dyn : Dynamics object
            the dynamics of the vehicle
        x0 : list or CasADi Object
            the initial state of the vehicle
        N : int
            the control horizon of the vehicle
        id : int
            the id of the vehicle
        color : str
            the color of the vehicle
        """
        x0 = cs.vertcat(x0)
        self.id = id
        self.reset_x0 = x0
        self.N = N
        self.dyn = car_dyn
        self.traj = Trajectory(N, car_dyn)
        self.traj.x0 = x0
        self.color = color
        self.default_u = [0] * self.dyn.nu * self.N

    def reset(self):
        """ Resets the initial state and the control actions of the vehicle """
        self.traj.x0 = self.reset_x0
        self.traj.u = self.default_u

    def move(self):
        """ Moves the vehicle by one time step """
        self.traj.x0 = self.dyn(self.traj.x0, self.traj.u)

    @property
    def x(self):
        return self.traj.x0

    @property
    def center(self):
        return self.center_x(self.x)

    def center_x(self, current_x):
        """ Returns the center of the vehicle, given the current state of the vehicle.

        Parameters
        ----------
        current_x : list
            the current state of the vehicle
         """
        # if isinstance(current_x[0], cs.SX) or isinstance(current_x[0], cs.MX):
        center_x = cs.vertcat(current_x[0] + (1 / 2) * (self.dyn.lf - self.dyn.lr) * cs.cos(current_x[2]),
                            current_x[1] + (1 / 2) * (self.dyn.lf - self.dyn.lr) * cs.sin(current_x[2]),
                            current_x[2],
                            current_x[3])
        # else:
        #     center_x = [current_x[0] + (1 / 2) * (self.dyn.lf - self.dyn.lr) * cs.cos(current_x[2]),
        #                         current_x[1] + (1 / 2) * (self.dyn.lf - self.dyn.lr) * cs.sin(current_x[2]),
        #                         current_x[2],
        #                         current_x[3]]
        return center_x

    @property
    def corners(self):
        return self.corners_x(self.x)

    def corners_x(self, current_x):
        """ Returns the four corners of the vehicle, given the current state of the vehicle.

        Parameters
        ----------
        current_x : list
            the current state of the vehicle
         """
        # if isinstance(current_x[0], cs.SX) or isinstance(current_x[0], cs.MX):
        four_corners = [cs.vertcat(current_x[0] + self.lf * cs.cos(current_x[2]) - self.width / 2. * cs.sin(current_x[2]), current_x[1] + self.lf * cs.sin(current_x[2]) + self.width / 2. * cs.cos(current_x[2])),
                    cs.vertcat(current_x[0] + self.lf * cs.cos(current_x[2]) + self.width / 2. * cs.sin(current_x[2]), current_x[1] + self.lf * cs.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])),
                    cs.vertcat(current_x[0] - self.lr * cs.cos(current_x[2]) + self.width / 2. * cs.sin(current_x[2]), current_x[1] - self.lr * cs.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])),
                    cs.vertcat(current_x[0] - self.lr * cs.cos(current_x[2]) - self.width / 2. * cs.sin(current_x[2]), current_x[1] - self.lr * cs.sin(current_x[2]) + self.width / 2. * cs.cos(current_x[2]))]
        # else:
        #     four_corners = [[current_x[0] + self.lf * np.cos(current_x[2]) - self.width / 2. * np.sin(current_x[2]), current_x[1] + self.lf * np.sin(current_x[2]) + self.width / 2. * np.cos(current_x[2])],
        #                [current_x[0] + self.lf * np.cos(current_x[2]) + self.width / 2. * np.sin(current_x[2]), current_x[1] + self.lf * np.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])],
        #                [current_x[0] - self.lr * np.cos(current_x[2]) + self.width / 2. * np.sin(current_x[2]), current_x[1] - self.lr * np.sin(current_x[2]) - self.width / 2. * np.cos(current_x[2])],
        #                [current_x[0] - self.lr * np.cos(current_x[2]) - self.width / 2. * np.sin(current_x[2]), current_x[1] - self.lr * np.sin(current_x[2]) + self.width / 2. * np.cos(current_x[2])]]
        return four_corners

    def front_x(self, current_x):
        """ Returns the front of the vehicle, given the current state of the vehicle.

        Parameters
        ----------
        current_x : list
            the current state of the vehicle
         """
        if self.lf == 0:
            return current_x[0:2]
        else:
            return cs.vertcat(current_x[0] + self.lf * cs.cos(current_x[2]), current_x[1] + self.lf * cs.sin(current_x[2]))
        return four_corners

    @property
    def lr(self):
        return self.dyn.lr

    @property
    def lf(self):
        return self.dyn.lf

    @property
    def len(self):
        return self.dyn.lf + self.dyn.lr

    @property
    def width(self):
        return self.dyn.width

    @property
    def u(self):
        return self.traj.u[0:self.dyn.nu]

    @u.setter
    def u(self, value):
        self.traj.u[0:self.dyn.nu] = value

    @property
    def u_2D(self):
        if self.dyn.nu == 1:
            return [self.traj.u[0], 0]
        else:
            return self.traj.u[0:self.dyn.nu]

    def control(self, steer, gas):
        pass

    def update(self, k):
        return 

    def get_future_trajectory(self):
        return self.traj.get_future_trajectory()

class UserControlledCar(Car):
    """
        A class used to represent a user controlled vehicle

        Attributes
        ----------
        _fixed_control : list
            the fixed control action of the vehicle

        Properties
        ----------
        fixed_control : list
            the fixed control action of the vehicle
        """
    def __init__(self, *args, **vargs):
        """
        Parameters
        ----------
        *args : arguments for Car object
        **vargs : optional arguments for Car object
        """
        Car.__init__(self, *args, **vargs)
        self._fixed_control = None

    @property
    def fixed_control(self):
        return self._fixed_control

    def fix_control(self, ctrl):
        """ Sets the value of the fixed control of the vehicle

        Parameters
        ----------
        ctrl : list
            the fixed control action of the vehicle
        """
        self._fixed_control = ctrl

    def control(self, steer, gas):
        """ Sets the value of the control action of the vehicle

        Parameters
        ----------
        steer : float
            the current steering angle of the vehicle
        gas: float
            the current acceleration of the vehicle
        """
        if self._fixed_control is not None:
            self.u = self._fixed_control
        if self.dyn.nu == 1:
            self.u = [gas]
        elif self.dyn.nu == 2:
            self.u = [steer, gas]


class GPGOptimizerCar(Car):
    """
    A class used to represent a vehicle solving a GPG formulation

    Properties
    ----------
    reward : Feature object
        the reward of the vehicle
    terminal_reward
        the terminal reward of the vehicle
    mode : str
        the shooting mode of the solver, i.e. 'single' or 'multiple'
    solver : str
        the solver for for inner problems in the decomposition method, i.e. 'ipopt' or 'OpEn' (or 'qpoases')
    symbolics_type : str
        the symbolics type for the problem, i.e. 'SX' or 'MX'
    """
    def __init__(self, *args, gpg_solver_settings=None, online_learning_settings=None):
        """
        Parameters
        ----------
        gpg_solver_settings : GPGSolverSettings object, optional
            the settings for the GPG solver
        online_learning_settings : OnlineLearningSettings object, optional
            the online learning settings
        """
        Car.__init__(self, *args)

        # Solver settings
        if gpg_solver_settings is None:
            gpg_solver_settings = settings.GPGSolverSettings()
        if online_learning_settings is None:
            online_learning_settings = settings.OnlineLearningSettings()
        self.online_learning_settings = online_learning_settings
        self.gpg_solver_settings = gpg_solver_settings

        # CasADi variable type
        self._sym = getattr(cs, 'SX')

        # Initialize optimizer and observer
        self.optimizer = None
        self.observer = None

        # Initialize players
        self.players = {}

        # SHARED COST
        self.stage_shared_reward = feature.empty()
        self.shared_reward = 0
        self.shared_reward_original = 0

        # SHARED CONSTRAINTS
        # Equality constraints
        self.g = {}
        self.g_original = {}
        self.stage_g = []
        self.soft_stage_g = []

        # Inequality constraints
        self.h = {}
        self.h_original = {}
        self.stage_h = []
        self.soft_stage_h = []
        self.terminal_h = []
        self.soft_terminal_h = []

        # Dual collision avoidance constraints
        self._stage_dual = []
        self._soft_stage_dual = []

        # Additional parameters and bounds
        self.v = {}
        self.lbv = {}
        self.ubv = {}
        self.z = {}
        self.lbz = {}
        self.ubz = {}

        self.x0_numeric_dict = {}
        self.p_numeric_dict = {}
        self.p_dict = {}
        self.nb_bounded = {}
        self.penalty_parameters = PenaltyParameters(self._sym, self.N, gpg_solver_settings)
        self.dyn_dict = {self.id: self.dyn}

        # Obstacles data
        self.obstacles = {}

        # Variables for storing obtained optimal values
        self.optimum_penalty_parameters = cs.DM()
        self.optimum_struct = {}
        self.optimum_mu_bounds_struct = {}
        self.optimum_mu_struct = {}

        # Solution times
        self.gpg_solution_time = 0
        self.observer_solution_time = 0

        # Deque for storing observations
        self._observations = deque([], online_learning_settings.nb_observations)

    def reset(self):
        Car.reset(self)

        # Reset initial parameter beliefs
        for player in self.players.values():
            player.reset_belief()

        # Reset observations deque
        self._observations = deque([], self.online_learning_settings.nb_observations)

        # Reset variables for storing obtained optimal values
        self.optimum_penalty_parameters = cs.DM()
        self.optimum_struct = {}
        self.optimum_mu_bounds_struct = {}
        self.optimum_mu_struct = {}

        # Reset controller and observer
        if self.optimizer is not None:
            self.optimizer.reset()
        if self.observer is not None:
            self.observer.reset()

    def move(self):
        """ Moves the vehicle by one time step """
        Car.move(self)
        self.traj.x0 = self.substitute_reward_params(self.traj.x0)

    def substitute_reward_params(self, state):
        if isinstance(state, cs.SX) or isinstance(state, cs.MX):
            return cs.DM(cs.substitute(state, self.ego.reward_params, self.ego.reward_params_current_belief)).toarray(True)
        else:
            return state

    def get_future_trajectory(self):
        future_list = self.traj.get_future_trajectory()
        future_list = [cs.DM(cs.substitute(state, self.ego.reward_params, self.ego.reward_params_current_belief)).toarray(True) for state in future_list]
        return future_list

    def get_human(self, i):
        """ Returns the human with the corresponding id

        Parameters
        ----------
        i : int
            the identifier of the human
        """
        return self.players[i].human

    def add_player(self, player, reward, terminal_reward=None, params=None, param_values=None):
        """ Adds a 'player', i.e. another GPGOptimizerCar, to the GPG formulation

        Parameters
        ----------
        player : GPGOptimizerCar object
            the added vehicle
        reward : Feature
            the stage reward of the added vehicle
        terminal_reward : Feature, optional
            the terminal reward of the added vehicle
        params : cs.SX or cs.MX, optional
            the cost function and constraint parameters
        param_values : list or CasADi Object, optional
            the initial estimate for the parameters
        """
        self.players[player.id] = Player(self.N, player, self._sym, reward, terminal_reward, params, param_values)

    def remove_player(self, key):
        self.players.pop(key)
        for dict in (self.g, self.g_original, self.h, self.h_original, self.v, self.lbv, self.ubv):
            to_remove = []
            for id_list in dict:
                if key in id_list:
                    to_remove.append(key)
            for key in to_remove:
                dict.pop(id_list)


    def add_player_g(self, i, g):
        """ Adds a player-specific stage equality constraint to the GPG formulation

        Parameters
        ----------
        i : int
            the identifier of the player
        g : Constraints object
            the equality constraints
        """
        self.players[i].player_stage_g.append(g)
        return

    def add_player_h(self, i, h):
        """ Adds a player-specific stage inequality constraint to the GPG formulation

        Parameters
        ----------
        i : int
            the identifier of the player
        h : Constraints object
            the inequality constraints
        """
        self.players[i].player_stage_h.append(h)
        return
    
    def add_player_terminal_h(self, i, h):
        """ Adds a player-specific terminal inequality constraint to the GPG formulation

        Parameters
        ----------
        i : int
            the identifier of the player
        h : Constraints object
            the inequality constraints
        """
        self.players[i].player_terminal_h.append(h)
        return

    def add_g(self, g):
        """ Adds a shared stage equality constraint to the GPG formulation

        Parameters
        ----------
        g : Constraints object
            the shared equality constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self.stage_g.append(g)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self.soft_stage_g.append(g)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return

    def add_h(self, h):
        """ Adds a shared stage inequality constraint to the GPG formulation

        Parameters
        ----------
        h : Constraints object
            the shared inequality constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self.stage_h.append(h)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self.soft_stage_h.append(h)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return

    def add_terminal_h(self, h):
        """ Adds a shared terminal inequality constraint to the GPG formulation

        Parameters
        ----------
        h : Constraints object
            the shared inequality constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self.terminal_h.append(h)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self.soft_terminal_h.append(h)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return
    
    def remove_terminal_h(self, i):
        return

    def add_dual(self, dual):
        """ Adds shared stage dual constraints to the GPG formulation

        Parameters
        ----------
        dual : Constraints object
            the shared dual constraints
        """
        if self.gpg_solver_settings.constraint_mode == 'hard':
            self._stage_dual.append(dual)
        elif self.gpg_solver_settings.constraint_mode == 'soft':
            self._soft_stage_dual.append(dual)
        else:
            raise Exception('Given constraint mode is unknown: ' + str(self.gpg_solver_settings.constraint_mode))
        return

    def get_current_constraint_violation(self, x_dict, u):
        def eval(constraints, params):
            list = []
            for con in constraints:
                list.extend(con(*params))
            return cs.vertcat(*list)
        def eval_h(constraints, params):
            return -(cs.fmin(0.0, eval(constraints, params)))
        def eval_g(constraints, params):
            return cs.fabs(eval(constraints, params))
        return cs.mmax(cs.vertcat(eval_h(self.players[self.id].player_stage_h,(x_dict, u)), eval_h(self.stage_h,(x_dict, u)), eval_g(self.players[self.id].player_stage_g,(x_dict, u)), eval_g(self.stage_g,(x_dict, u))))

    @property
    def ego(self):
        return self.players[self.id]

    @property
    def reward(self):
        return self.ego._stage_reward

    def get_current_reward(self, *args):
        # try:
        #     return cs.DM(cs.substitute(self.reward(*args), self.ego.reward_params, self.ego.reward_params_current_belief))
        # except KeyError:
        #     return self.reward(*args)
        try:
            return cs.DM(cs.substitute(self.reward(*args) + self.stage_shared_reward(*args), self.ego.reward_params, self.ego.reward_params_current_belief))
        except KeyError:
            return self.reward(*args) + self.stage_shared_reward(*args)

    @property
    def terminal_reward(self):
        return self.ego._terminal_reward

    # @terminal_reward.setter
    # def terminal_reward(self, val):
    #     self._terminal_reward = val
    #     self.optimizer = None

    # def set_ego_params(self, params, param_values):
    #     if params is not None:
    #         self.p_dict[self.id] = params
    #         self.p_numeric_dict[self.id] = param_values

    def add_obstacle(self, vehicle):
        """ Adds an 'obstacle', i.e. another Car object with e.g. fixed motion to the GPG formulation

        Parameters
        ----------
        vehicle : Car object
            the added vehicle
        """
        self.obstacles[vehicle.id] = Obstacle(self.N, vehicle, self._sym)
        return

    @property
    def mode(self):
        return self.gpg_solver_settings.shooting_mode

    @property
    def solver(self):
        return self.gpg_solver_settings.solver

    @property
    def symbolics_type(self):
        return self._sym.type_name()

    @symbolics_type.setter
    def symbolics_type(self, val):
        assert(val == 'SX' or val == 'MX')
        self._sym = getattr(cs, val)

    def setup_x0_and_u(self):
        """ Initializes the CasADi symbolics for the initial states and control variables of the players in the GPG """
        for player in self.players.values():
            player.setup_x0_and_u()
        for obstacle in self.obstacles.values():
            obstacle.setup_x0_and_u()
        return

    def setup_x(self):
        """ Initializes list of state variables for the players in the GPG """
        for player in self.players.values():
            player.setup_x(self.mode)
        for obstacle in self.obstacles.values():
            obstacle.setup_x(self.mode)
        return

    def setup_rewards(self):
        """ Initializes the reward function for the players in the GPG """
        # others = self.players | self.obstacles  # Python 3.9+ only
        others = {**self.players, **self.obstacles}
        for player in self.players.values():
            player.setup_rewards(others)
        
        self.shared_reward = self.ego.trajectory.shared_reward(self.stage_shared_reward, others)
        self.shared_reward_original = self.shared_reward
        return

    def setup_constraints(self):
        """ Initializes the constraints for the players in the GPG

        When soft constraints are used, penalty parameters are introduced and the reward function adapted accordingly
        """
        # others = self.players | self.obstacles  # Python 3.9+ only
        others = {**self.players, **self.obstacles}

        def add_list_to_dict(dict, key, list):
            if key not in dict.keys():
                dict[key] = list
            else:
                dict[key].extend(list)

        def add_cs_to_dict(dict, key, variables):
            if key not in dict.keys():
                dict[key] = variables
            else:
                dict[key] = cs.vertcat(dict[key], variables)

        for con in self.stage_g:
            add_list_to_dict(self.g, con.id_list, self.traj.constraints(con, others))

        for con in self.stage_h:
            add_list_to_dict(self.h, con.id_list, self.traj.constraints(con, others))

        x_N = {}
        for id, object in others.items():
            x_N[id] = object.x[self.N-1]

        for con in self.terminal_h:
            add_list_to_dict(self.h, con.id_list, con(x_N))

        for player in self.players.values():
            player.setup_stage_constraints(others)
            player.setup_terminal_constraints(x_N)
        
        for con in self._stage_dual:
            eq, ineq, additional_parameters = self.traj.constraints_dual_formulation(con, others, self._sym)
            add_list_to_dict(self.g, con.id_list, eq)
            add_list_to_dict(self.h, con.id_list, ineq)
            add_cs_to_dict(self.v, con.id_list, additional_parameters)
            add_list_to_dict(self.lbv, con.id_list, [-float("inf")] * additional_parameters.numel())
            add_list_to_dict(self.ubv, con.id_list, [float("inf")] * additional_parameters.numel())

        self.h_original = self.h.copy()
        self.g_original = self.g.copy()
        self.shared_reward_original = self.shared_reward

        for con in self.soft_stage_g:
            current_constraint = self.traj.constraints(con, others)
            reward_contribution, z = self.penalty_parameters.add_penalty_constraints(current_constraint, 'stage_g')
            add_list_to_dict(self.g_original, con.id_list, current_constraint)
            self.shared_reward += reward_contribution

        for con in self.soft_stage_h:
            current_constraint = self.traj.constraints(con, others)
            reward_contribution, z = self.penalty_parameters.add_penalty_constraints(current_constraint, 'stage_h')
            add_list_to_dict(self.h_original, con.id_list, current_constraint)
            self.shared_reward += reward_contribution
            if z is not None:
                add_list_to_dict(self.h, con.id_list, [current_constraint[i] + z[i] for i in range(current_constraint.length)])
                add_cs_to_dict(self.z, con.id_list, z)
                add_list_to_dict(self.lbz, con.id_list, [0] * current_constraint.length)
                add_list_to_dict(self.ubz, con.id_list, [float("inf")] * current_constraint.length)

        for con in self.soft_terminal_h:
            current_constraint = con(x_N)
            reward_contribution, z = self.penalty_parameters.add_penalty_constraints(current_constraint, 'terminal_h')
            add_list_to_dict(self.h_original, con.id_list, current_constraint)
            self.shared_reward += reward_contribution
            if z is not None:
                add_list_to_dict(self.h, con.id_list, [current_constraint[i] + z[i] for i in range(current_constraint.length)])
                add_cs_to_dict(self.z, con.id_list, z)
                add_list_to_dict(self.lbz, con.id_list, [0] * current_constraint.length)
                add_list_to_dict(self.ubz, con.id_list, [float("inf")] * current_constraint.length)

        # for con in self._soft_stage_dual:  # TODO
        #     eq, ineq, additional_parameters = self.traj.constraints_dual_formulation(con, self.x_dict, self._sym)
        #     self.g_original.extend(eq)
        #     self.h_original.extend(ineq)
        #     self.v = cs.vertcat(self.v, additional_parameters)
        #     self.lbv += [-float("inf")] * additional_parameters.shape[0]
        #     self.ubv += [float("inf")] * additional_parameters.shape[0]

        return

    def setup_bounds(self):
        """ Initializes the bounds for the players in the GPG """
        for player in self.players.values():
            player.setup_bounds()
        return

    def setup_parameters(self):
        """ Initializes the parameters required for obstacles in the GPG """
        for obstacle_id, obstacle in self.obstacles.items():
            self.p_dict[obstacle_id] = cs.vertcat(obstacle.x0, obstacle.u)
        return

    def initialize_solvers(self):
        """ Initializes the GPG solver and the online learning methodology of the ego vehicle """
        self.setup_x0_and_u()
        self.setup_x()
        self.setup_rewards()
        self.setup_constraints()
        self.setup_parameters()
        self.setup_bounds()

        if self.gpg_solver_settings.use_gauss_seidel:
            self.optimizer = gaussseidelsolver.solver(self.id, self.players, self.shared_reward, self.g, self.h, self.v, self.lbv, self.ubv,
                                                         self.p_dict, self.penalty_parameters, self._sym, self.gpg_solver_settings)
        else:
            self.optimizer = lagrangiansolver.solver(self.id, self.players, self.shared_reward, self.g, self.h, self.v, self.lbv, self.ubv,
                                                         self.p_dict, self.penalty_parameters, self.gpg_solver_settings)
        if self.online_learning_settings.based_on_original_gpg:
            self.observer = learning.online_learning_solver(self.id, self.players, self.shared_reward_original, self.g_original, self.h_original, self._sym(),
                                                            self.v, self.lbv, self.ubv, self.p_dict, self._sym, self.online_learning_settings)
        else:
            bool_g_quad = bool(self.g) and not self.gpg_solver_settings.panoc_use_alm and self.gpg_solver_settings.solver != 'ipopt'

            #     self.observer = learning.online_learning_solver_full(self.id, self.players, self.shared_reward + shared_penalty_reward, {}, self.h,
            #                                                     cs.vertcat(self.penalty_parameters.all, shared_penalty_parameters),
            #                                                     self.v, self.lbv, self.ubv, self.p_dict, self._sym, self.online_learning_settings)
            # else:
            self.observer = learning.online_learning_solver_full(self.id, self.players, self.shared_reward, self.g, self.h, self.penalty_parameters.all,
                                                                self.v, self.lbv, self.ubv, self.p_dict, self._sym, self.online_learning_settings, bool_g_quad)

    def control(self, steer, gas):
        """ Sets the value of the control action of the ego vehicle by solving a GPG formulation using the current
        belief in the parameters of the human drivers """
        # Initialize optimizers
        if self.optimizer is None:
            self.initialize_solvers()

        # Initialize current states and parameters
        for id_player, player in self.players.items():
            self.x0_numeric_dict[id_player] = player.vehicle.x
        for id_obstacle, obstacle in self.obstacles.items():
            self.p_numeric_dict[id_obstacle] = cs.vertcat(obstacle.vehicle.x,cs.repmat(obstacle.vehicle.u, self.N))

        # Solve the GPG
        start = time.perf_counter()
        self.optimum_struct, self.optimum_mu_bounds_struct, self.optimum_mu_struct, self.optimum_penalty_parameters =\
            self.optimizer.minimize(self.x0_numeric_dict, self.players, self.p_numeric_dict)
        end = time.perf_counter()
        self.gpg_solution_time = (end-start)
        self.traj.u = self.optimum_struct[str(self.id)][0:self.N*self.dyn.nu]
        print('id: ' + str(self.id) + ', u_opt: ' + str(self.optimum_struct[str(self.id)]))
        print('Solution time: ' + str(self.gpg_solution_time) + ', CPU time: ' + str(self.optimizer.cpu_time))

    def observe(self):
        """ Solves the online learning problem, updating the belief in the parameters of the 'human' players"""
        if self.online_learning_settings.nb_observations > 0:
            # Observe the human actions
            observed_actions = {}
            for i, player in self.players.items():
                observed_actions[i] = player.vehicle.u

            if self.online_learning_settings.based_on_original_gpg:
                self.optimum_mu_bounds_struct['common'] = self.optimum_mu_bounds_struct['common'][-self.penalty_parameters.nb_z:] #TODO does this work? Test exact penalty
                self.optimum_struct['common'] = self.optimum_struct['common'][-self.penalty_parameters.nb_z:]
                self._observations.appendleft(
                    [self.x0_numeric_dict.copy(), self.optimum_struct,
                     observed_actions, self.optimum_mu_bounds_struct, self.optimum_mu_struct,
                     self.p_numeric_dict.copy(), cs.DM()])
            else:
                self._observations.appendleft(
                    [self.x0_numeric_dict.copy(), self.optimum_struct, observed_actions,
                     self.optimum_mu_bounds_struct, self.optimum_mu_struct, self.p_numeric_dict.copy(),
                     self.optimum_penalty_parameters])

            # if self.online_learning_settings.based_on_original_gpg:
            #     optimum_mu_bounds_struct_copy = self.optimum_mu_bounds_struct.copy() 
            #     optimum_mu_bounds_struct_copy['common'] = optimum_mu_bounds_struct_copy['common'][-self.penalty_parameters.nb_z:]
            #     optimum_struct_copy = self.optimum_struct.copy()
            #     optimum_struct_copy['common'] = optimum_struct_copy['common'][-self.penalty_parameters.nb_z:]
            #     self._observations.appendleft(
            #         [self.x0_numeric_dict.copy(), optimum_struct_copy,
            #          observed_actions, optimum_mu_bounds_struct_copy, self.optimum_mu_struct.copy(),
            #          self.p_numeric_dict.copy(), cs.DM()])
            # else:
            #     self._observations.appendleft(
            #         [self.x0_numeric_dict.copy(), self.optimum_struct.copy(), observed_actions,
            #          self.optimum_mu_bounds_struct.copy(), self.optimum_mu_struct.copy(), self.p_numeric_dict.copy(),
            #          self.optimum_penalty_parameters])

            # Solve the online learning methodology
            start = time.perf_counter()
            self.observer.observe(self.players, self._observations)
            end = time.perf_counter()
            self.observer_solution_time = end - start
            print('Observation time: ' + str(self.observer_solution_time) + ', CPU time: ' + str(self.observer.cpu_time))