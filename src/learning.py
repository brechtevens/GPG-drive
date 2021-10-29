import numpy as np
import casadi as cs
import opengen as og
import os, sys
import src.solvers
from src.settings import OnlineLearningSettings

from itertools import chain

import casadi.tools as ct

def make_bounds(players, g, h, lbv, ubv):
    f_lb = lambda player: player.lb
    f_ub = lambda player: player.ub
    f_lbg = lambda player: player.lbg
    f_ubg = lambda player: player.ubg
    f_lbh = lambda player: player.lbh
    f_ubh = lambda player: player.ubh

    lbg_common = [0] * len(g)
    ubg_common = [0] * len(g)
    lbh_common = [0] * len(h)
    ubh_common = [float('inf')] * len(h)

    bounds = {
        'lbx': list(chain.from_iterable((f_lb(player)) for player in players.values())),
        'ubx': list(chain.from_iterable((f_ub(player)) for player in players.values())),
        'lbg': lbg_common + list(chain.from_iterable((f_lbg(player)) for player in players.values())),
        'ubg': ubg_common + list(chain.from_iterable((f_ubg(player)) for player in players.values())),
        'lbh': list(chain.from_iterable((f_lbh(player)) for player in players.values())) + lbh_common,
        'ubh': list(chain.from_iterable((f_ubh(player)) for player in players.values())) + ubh_common
    }
    bounds['lbx'].extend(lbv)
    bounds['ubx'].extend(ubv)
    return bounds


def make_full_bounds(players, g, h, lbv, ubv, nb_params, nb_lambdas, nb_mus, nb_mu_bounds, nb_observations):
    f_lb = lambda player: player.lb[player.trajectory.dyn.nu:]
    f_ub = lambda player: player.ub[player.trajectory.dyn.nu:]

    bounds = {
        'lbx': [0] * nb_params + list(chain.from_iterable((f_lb(player)) for player in players.values())) * nb_observations + \
               [-float('inf')] * nb_lambdas + [0] * nb_mus,
        'ubx': [float('inf')] * nb_params + list(chain.from_iterable((f_ub(player)) for player in players.values())) * nb_observations + \
               [float('inf')] * (nb_lambdas + nb_mus),
        'lbg' : [0] * (nb_lambdas + nb_mus) + [0] * (nb_mus - nb_mu_bounds),
        'ubg' : [0] * (nb_lambdas + nb_mus) + [float("inf")] * (nb_mus - nb_mu_bounds),
        'lbg1' : [0] * (nb_lambdas) + [0] * (nb_mus - nb_mu_bounds),
        'ubg1' : [0] * (nb_lambdas) + [float("inf")] * (nb_mus - nb_mu_bounds),
        'lbg2' : [0] * (nb_mus),
        'ubg2' : [0] * (nb_mus)
    }
    bounds['lbx'].extend(lbv)
    bounds['ubx'].extend(ubv)
    return bounds


class online_learning_solver(object):
    """
    A class used to set-up and solve the proposed online learning methodology
    """

    # def __init__(self, ego_id, f_dict, u_dict, x0_dict, lbx_dict, ubx_dict, g, h, player_g, player_h, penalty_parameters,
    #              v, theta_dict, p_dict, symbolics, nu, settings, x_dict=None, g_dyn_dict=None):
    def __init__(self, ego_id, players, shared_reward, g_dict, h_dict, penalty_parameters,
                 v_dict, lbv_dict, ubv_dict, p_dict, symbolics, settings:OnlineLearningSettings):
        """
        Parameters
        ----------
        ego_id : int
            the id of the ego vehicle
        players : dict
            dictionary consisting of the players in the GPG
        shared_reward
            shared reward for all players
        g_dict : dict
            the shared equality constraints
        h_dict : dict
            the shared inequality constraints
        penalty_parameters : [cs.SX, cs.SX] or [cs.MX, cs.MX]
            the penalty parameters of the GPG
        v_dict : dict
            the additional shared optimization variables
        lbv_dict : dict
            the lower bounds on the additional shared optimization variables
        ubv_dict : dict
            the upper bounds on the additional shared optimization variables
        p_dict : dict
            the parameters of the obstacles in the GPG or any additional parameters
        symbolics: cs.SX or cs.MX attribute
            allows to generate additional SX or MX symbolics
        settings : OnlineLearningSettings object
            the settings for the online learning methodology
        """
        # Do not generate observer if nb_observations == 0
        self.cpu_time = 0
        if settings.nb_observations == 0:
            return

        # Store variables
        self.id = ego_id
        self.settings:OnlineLearningSettings = settings
        self.symbolics = symbolics

        # Initialize other variables
        self.x = symbolics()

        def proper_bounds(lower_bound, upper_bound):
            if not (lower_bound == -float('inf') and upper_bound == float('inf')):
                return True
            return False

        # Initialize dictionaries for optimization variables, online learning problems and their solvers
        self.observation_variables = {}
        self.observation_problem_dict = {}
        self.observation_solver_dict = {}
        self.bounds_dict = {}

        # Initialize blocks of variables
        self.block_indices = {}
        self.block_keys = {}

        g = tuple(*g_dict.values())
        h = tuple(*h_dict.values())
        lbv = tuple(*lbv_dict.values())
        ubv = tuple(*ubv_dict.values())
        v = cs.vertcat(*v_dict.values())
        self.nb_v = v.numel()

        # Set-up the symbolics, Lagrangian, constraints etc for the online learning methodology
        for id, player in players.items():
            if id != ego_id:
                # The cost function of the observer
                V_full = symbolics()
                # The parameter vector of the observer
                p_full = symbolics()
                # The penalty vector of the observer
                penalty_full = symbolics()
                # The vector containing the optimization variables (v denotes the 'additional' optimization variables)
                x_v_full = symbolics()
                # The full vector of equality constraints for the observer
                g_full = symbolics()
                # The full vector of inequality constraints for the observer
                h_full = symbolics()
                # The full vector of inequality constraints for the observer corresponding to parametric rectangle bounds
                h_full_bounds = symbolics()
                # The full vector of Lagrangian parameters mu for the observer
                mu_full = symbolics()
                # The full vector of mu times h for the observer
                mu_times_h_full = symbolics()
                # The full vector of mu times h for the observer
                mu_times_h_full_bounds = symbolics()
                # The full vector of Lagrangian parameters lambda for the observer
                lambda_full = symbolics()

                # Make lists for x0, x and the parameters
                x0_list = [p.x0 for p in players.values()]
                x_list = [p.opt for p in players.values()]
                parameters_list = [p_dict[key] for key in p_dict]
                nb_mu_bounds = 0

                # Iterate over the number of previous observations used to update the parameter
                for t in range(settings.nb_observations):
                    # Make CasADi symbolics for optimization variables and parameters at this time stamp
                    x0_step = symbolics()
                    x_step = symbolics()
                    v_step = symbolics()
                    parameters_step = symbolics()
                    p_step = symbolics()
                    u_step = symbolics()
                    lambda_step = symbolics.sym('lambda_step_' + str(t), len(g) + player.nb_player_g, 1)
                    mu_step = symbolics.sym('mu_step_' + str(t), len(h) + player.nb_player_h, 1)

                    # Initialize the box constraints
                    def get_bound(lower, upper, var):
                        try:
                            if lower == -float('inf'):
                                return upper - var
                        except RuntimeError:
                            pass
                        try:
                            if upper == float('inf'):
                                return var - lower
                        except RuntimeError:
                            pass
                        return (upper - var) * (var - lower)

                    mu_step_bounds = symbolics.sym('', 0, 1)
                    h_step_bounds = symbolics.sym('', 0, 1)

                    # Define box constraints for optimizations variables
                    for index, bounds in enumerate(zip(player.lb, player.ub)):
                        lower_bound, upper_bound = bounds
                        if proper_bounds(lower_bound, upper_bound):
                            mu_step_bounds = cs.vertcat(mu_step_bounds, symbolics.sym(
                                'mu_step_bounds_' + str(t), 1, 1))
                            bound = get_bound(lower_bound, upper_bound, player.opt[index])
                            h_step_bounds = cs.vertcat(h_step_bounds, bound)

                    # Define box constraints for shared optimizations variables v
                    for index, bounds in enumerate(zip(lbv, ubv)):
                        lower_bound, upper_bound = bounds
                        if proper_bounds(lower_bound, upper_bound):
                            mu_step_bounds = cs.vertcat(mu_step_bounds, symbolics.sym(
                                'mu_step_bounds_' + str(t), 1, 1))
                            bound = get_bound(lower_bound, upper_bound, v[index])
                            h_step_bounds = cs.vertcat(h_step_bounds, bound)

                    # Set-up Lagrangian at the current time stamp
                    cost = - shared_reward - player.reward
                    if len(g) + player.nb_player_g > 0:
                        cost -= cs.dot(lambda_step, cs.vertcat(*g, *player.player_g))
                    if len(h) + player.nb_player_h > 0:
                        cost -= cs.dot(mu_step, cs.vertcat(*h, *player.player_h))
                    cost -= cs.sum1(mu_step_bounds * h_step_bounds)
                    lagrangian = cs.jacobian(cost, cs.vertcat(player.opt, v))

                    # Set-up x0 variables
                    for i, p in players.items():
                        sym = symbolics.sym('x0_' + str(i) + '_step_' + str(t), p.x0.shape)
                        x0_step = cs.vertcat(x0_step, sym)
                        p_step = cs.vertcat(p_step, sym)

                    # Set-up x variables
                    for i, p in players.items():
                        if i != id:
                            sym = symbolics.sym('x_' + str(i) + '_step_' + str(t), p.opt.shape)
                            x_step = cs.vertcat(x_step, sym)
                            p_step = cs.vertcat(p_step, sym)
                        else:
                            # For the regarded player, the control variables at the first time step are observed
                            sym_p = symbolics.sym('x_' + str(i) + '_step_' + str(t), p.trajectory.dyn.nu, 1)
                            sym_u = symbolics.sym('x_' + str(i) + '_step_' + str(t), p.opt.numel() - p.trajectory.dyn.nu, 1)
                            x_step = cs.vertcat(x_step, sym_p, sym_u)
                            p_step = cs.vertcat(p_step, sym_p)
                            u_step = sym_u

                    # Set-up parameters
                    for i in p_dict:
                        sym = symbolics.sym('x0_' + str(i) + '_step_' + str(t), p_dict[i].shape)
                        parameters_step = cs.vertcat(parameters_step, sym)
                        p_step = cs.vertcat(p_step, sym)

                    # Set-up penalty parameters
                    penalty_step = symbolics.sym('rho_step_' + str(t), penalty_parameters.shape)

                    # Set-up additional optimization variables
                    v_step = cs.vertcat(v_step, symbolics.sym('v_step_' + str(t), v.shape))

                    # Substitute Lagrangian and constraints with new optimization variables
                    V_step = cs.substitute(lagrangian,
                                        cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                        cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
                    g_step = cs.substitute(cs.vertcat(*g, *player.player_g),
                                        cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                        cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
                    h_step = cs.substitute(cs.vertcat(*h, *player.player_h),
                                        cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                        cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
                    h_step_bounds = cs.substitute(h_step_bounds,
                                                cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                                cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))

                    # Add new variables to the full vectors storing this information
                    x_v_full = cs.vertcat(x_v_full, u_step, v_step)
                    V_full = cs.horzcat(V_full, V_step)
                    g_full = cs.vertcat(g_full, g_step)
                    h_full = cs.vertcat(h_full, h_step)
                    # h_full_bounds = cs.vertcat(h_full_bounds, h_step_bounds[self.parametric_bounds_index_dict[id]])
                    p_full = cs.vertcat(p_full, p_step)
                    penalty_full = cs.vertcat(penalty_full, penalty_step)
                    mu_full = cs.vertcat(mu_full, mu_step, mu_step_bounds)
                    nb_mu_bounds += mu_step_bounds.numel()
                    if h_step.numel() > 0:
                        mu_times_h_full = cs.vertcat(mu_times_h_full, mu_step * h_step)
                    if h_step_bounds.numel() > 0:
                        mu_times_h_full_bounds = cs.vertcat(mu_times_h_full_bounds, mu_step_bounds * h_step_bounds)
                    lambda_full = cs.vertcat(lambda_full, lambda_step)

                # Add regularization term
                self.two_norm_lagrangian = cs.sumsqr(V_full)
                current_estimate_parameters = symbolics.sym('current_estimate_parameters', player.nb_reward_params, 1)
                try:
                    regularization = symbolics.sym('reg', len(settings.regularization))
                except:
                    regularization = symbolics.sym('reg')
                V_full = cs.sumsqr(V_full) + cs.sum1(regularization * (player.reward_params - current_estimate_parameters) ** 2)

                # Set up parameter vector
                p_full = cs.vertcat(p_full, penalty_full, current_estimate_parameters, regularization)

                # Store all required information in a single dictionary
                self.observation_variables[id] = {'theta': player.reward_params, 'x_v': x_v_full, 'lambda': lambda_full,
                                                'mu': mu_full, 'f': V_full, 'p': p_full, 'g': g_full, 'h': h_full,
                                                'h_bounds': h_full_bounds, 'mu_times_h': mu_times_h_full,
                                                'mu_times_h_bounds': mu_times_h_full_bounds}

                self.observation_variables[id]['x'] = cs.vertcat(self.observation_variables[id]['theta'], self.observation_variables[id]['x_v'],
                                    self.observation_variables[id]['lambda'], self.observation_variables[id]['mu'])

                nb_lambdas = self.observation_variables[id]['lambda'].numel()
                nb_mus = self.observation_variables[id]['mu'].numel()

                self.bounds_dict[id] = {
                    'lbx' : [0] * player.nb_reward_params + player.lb[player.trajectory.dyn.nu:] * settings.nb_observations + \
                                    [-float('inf')] * nb_lambdas + [0] * nb_mus,
                    'ubx' : [float('inf')] * player.nb_reward_params + player.ub[player.trajectory.dyn.nu:] * settings.nb_observations + \
                                    [float('inf')] * (nb_lambdas + nb_mus),
                    'lbh' : [0] * (nb_mus - nb_mu_bounds),
                    'ubh' : [float("inf")] * (nb_mus - nb_mu_bounds),
                    'lbg' : [0] * (nb_lambdas + nb_mus),
                    'ubg' : [0] * (nb_lambdas + nb_mus)
                }

        # Set up the optimization problem
        self.setup_solver(players)

    def reset(self):
        return

    def setup_solver(self, players):
        """ Builds the solver for the online learning methodology for each human player with unknown parameters """
        print("Build Learner")
        for id, player in players.items():
            if id != self.id:
                keys = ['theta', 'x_v', 'lambda', 'mu']
                current_index = 0
                self.block_indices[id] = {}
                self.block_keys[id] = []
                for key in keys:
                    sx = self.observation_variables[id][key]
                    if sx.numel() > 0:
                        new_index = current_index + sx.numel()
                        self.block_indices[id][key] = (current_index, new_index)
                        self.block_keys[id].append(key)
                        current_index = new_index

                g = cs.vertcat(self.observation_variables[id]['g'], self.observation_variables[id]['mu_times_h'],
                    self.observation_variables[id]['mu_times_h_bounds'], self.observation_variables[id]['h'])
                g1 = cs.vertcat(self.observation_variables[id]['g'], self.observation_variables[id]['h'])
                g2 = cs.vertcat(self.observation_variables[id]['mu_times_h'], self.observation_variables[id]['mu_times_h_bounds'])

                self.observation_problem_dict[id] = {'x': self.observation_variables[id]['x'],
                                                    'f': self.observation_variables[id]['f'],
                                                    'p': self.observation_variables[id]['p'],
                                                    'g': g, 'g1': g1, 'g2': g2}

                # set-up solver for online learning methodology
                if self.settings.solver == 'ipopt':
                    self.observation_solver_dict[id] = src.solvers.get_ipopt_solver(self.observation_problem_dict[id], self.settings, self.bounds_dict[id])

                else:
                    self.primal_factor = self.settings.panoc_delta_tolerance / self.settings.panoc_delta_tolerance_primal_feas
                    self.complementarity_factor = self.settings.panoc_delta_tolerance / self.settings.panoc_delta_tolerance_complementarity
                    self.complementarity_bounds_factor = self.settings.panoc_delta_tolerance / self.settings.panoc_delta_tolerance_complementarity_bounds
                    factor = self.symbolics.sym('factor', 3)
                    self.observation_problem_dict[id]['p'] = cs.vertcat(self.observation_variables[id]['p'], factor)
                    self.observation_problem_dict[id]['g'] = cs.vertcat(
                                factor[0]*self.observation_variables[id]['g'],
                                factor[0]*self.observation_variables[id]['mu_times_h'],
                                factor[1]*self.observation_variables[id]['mu_times_h_bounds'],
                                factor[2]*self.observation_variables[id]['h'])
                    self.observation_problem_dict[id]['g1'] = cs.vertcat(
                                factor[0]*self.observation_variables[id]['g'],
                                factor[2]*self.observation_variables[id]['h'])
                    self.observation_problem_dict[id]['g2'] = cs.vertcat(
                                factor[0]*self.observation_variables[id]['mu_times_h'],
                                factor[1]*self.observation_variables[id]['mu_times_h_bounds'])
                    
                    if self.settings.solver == 'OpEn':
                        self.observation_solver_dict[id] = src.solvers.get_OpEn_solver(self.observation_problem_dict[id], self.settings, self.bounds_dict[id], self.id, id, "learn")
                    else:
                        self.observation_solver_dict[id] = src.solvers.get_panocpy_solver(self.observation_problem_dict[id], self.settings, self.bounds_dict[id], self.id, id, "learn")

    def observe(self, players, observations):
        """ Initializes the bounds for the players in the GPG

        Parameters
        ----------
        players : dict
            the current estimate for the parameters of the human players
        observations : deque object
            contains the required information for initializing and warm-starting the online learning methodology
        """
        if observations.maxlen == 0 or len(observations) < observations.maxlen:
            return
        # new_estimate_parameters_dict = {}

        # Set-up optimization variables and parameter lists in the correct form based on the given observation object
        for id, player in players.items():
            if id != self.id:
                p_numeric = []
                penalty_numeric = []
                x_v0_numeric = []
                mu_bounds_numeric = []
                mu_numeric = []
                lambda_numeric = []
                for observation in observations:
                    p_step = []
                    u_step = []
                    v_step = []
                    mu_bounds_numeric_step = []
                    mu_numeric_step = []
                    lambda_numeric_step = []
                    if len(observation) == 7:
                        for i, value in observation[0].items():
                            p_step = cs.vertcat(p_step, value)
                        for i in observation[1].keys():
                            if i == 'common':
                                v_step = observation[1][i]
                            elif int(i) != id:
                                p_step = cs.vertcat(p_step, observation[1][i])
                            else:
                                p_step = cs.vertcat(p_step, observation[2][int(i)][:player.trajectory.dyn.nu])
                                u_step = observation[1][i][player.trajectory.dyn.nu:]
                                mu_bounds_numeric_step = observation[3][str(i)]
                                mu_numeric_step = cs.vertcat(observation[4]['g_common'], observation[4]['g_' + str(i)]) #TODO: observation[4].cat?
                                lambda_numeric_step = cs.vertcat(observation[4]['h_' + str(i)], observation[4]['h_common'])
                        for i, value in observation[5].items():
                            p_step = cs.vertcat(p_step, value)
                        penalty_step = observation[6]

                        p_numeric = cs.vertcat(p_numeric, p_step)
                        penalty_numeric = cs.vertcat(penalty_numeric, penalty_step)
                        x_v0_numeric = cs.vertcat(x_v0_numeric, u_step, v_step)
                        mu_bounds_numeric = cs.vertcat(mu_bounds_numeric, mu_bounds_numeric_step)
                        mu_numeric = cs.vertcat(mu_numeric, mu_numeric_step)
                        lambda_numeric = cs.vertcat(lambda_numeric, lambda_numeric_step)
                    else:
                        raise Exception('Observation format is incorrect')
                if cs.vertcat(mu_numeric, mu_bounds_numeric).numel() == 0:
                    nb_lambdas = self.observation_variables[id]['lambda'].numel()
                    nb_mus = self.observation_variables[id]['mu'].numel()
                    lambda_numeric = cs.DM.zeros(nb_lambdas, 1)
                    mu_numeric = cs.DM.zeros(nb_mus, 1)
                    mu_bounds_numeric = cs.DM.zeros(0, 1)
                p_numeric = cs.vertcat(p_numeric, penalty_numeric, player.reward_params_current_belief, self.settings.regularization)
                x = cs.vertcat(player.reward_params_current_belief, x_v0_numeric, lambda_numeric, mu_numeric, mu_bounds_numeric)
                # Solve online learning optimization problem
                if self.settings.solver == 'ipopt':
                    solution, self.cpu_time = self.observation_solver_dict[id](x, p_numeric)
                    x = solution['x']
                    a, b = self.block_indices[id]['theta']
                    player.reward_params_current_belief = list(x[a:b].toarray(True))
                elif self.settings.solver == 'OpEn':
                    solution = self.observation_solver_dict[id](x.toarray(True),cs.vertcat(p_numeric, self.complementarity_factor,
                                                                                           self.complementarity_bounds_factor, self.primal_factor).toarray(True))
                    self.cpu_time = solution.solve_time_ms / 1000
                    x = solution.solution
                    a, b = self.block_indices[id]['theta']
                    player.reward_params_current_belief = x[a:b]
                else:
                    mu, x, stats = self.observation_solver_dict[id](x.toarray(True),cs.vertcat(p_numeric, self.complementarity_factor,
                                                                                              self.complementarity_bounds_factor, self.primal_factor).toarray(True))
                    self.cpu_time = stats['elapsed_time'].total_seconds()
                    a, b = self.block_indices[id]['theta']
                    player.reward_params_current_belief = x[a:b]

                print('Online learning:  2-norm of Lagrangian: ' + str(
                    cs.substitute(self.two_norm_lagrangian,
                                cs.vertcat(self.observation_variables[id]['x'], self.observation_variables[id]['p']),
                                cs.vertcat(x, p_numeric))))
                print('New estimate parameters = ' + str(player.reward_params_current_belief))
        return

class online_learning_solver_full(object):
    """
    A class used to set-up and solve the proposed online learning methodology
    """

    # def __init__(self, ego_id, f_dict, u_dict, x0_dict, lbx_dict, ubx_dict, g, h, player_g, player_h, penalty_parameters,
    #              v, theta_dict, p_dict, symbolics, nu, settings, x_dict=None, g_dyn_dict=None):
    def __init__(self, ego_id, players, shared_reward, g_dict, h_dict, penalty_parameters,
                 v_dict, lbv_dict, ubv_dict, p_dict, symbolics, settings:OnlineLearningSettings, bool_g_quad):
        """
        Parameters
        ----------
        ego_id : int
            the id of the ego vehicle
        players : dict
            dictionary consisting of the players in the GPG
        shared_reward
            shared reward for all players
        g_dict : dict
            the shared equality constraints
        h_dict : dict
            the shared inequality constraints
        penalty_parameters : [cs.SX, cs.SX] or [cs.MX, cs.MX]
            the penalty parameters of the GPG
        v_dict : dict
            the additional shared optimization variables
        lbv_dict : dict
            the lower bounds on the additional shared optimization variables
        ubv_dict : dict
            the upper bounds on the additional shared optimization variables
        p_dict : dict
            the parameters of the obstacles in the GPG or any additional parameters
        symbolics: cs.SX or cs.MX attribute
            allows to generate additional SX or MX symbolics
        settings : OnlineLearningSettings object
            the settings for the online learning methodology
        bool_g_quad : boolean
            whether g_dict is handled using quadratic penalty or not
        """
        # Do not generate observer if nb_observations == 0
        self.cpu_time = 0
        if settings.nb_observations == 0:
            return

        # Store variables
        self.id = ego_id
        self.settings:OnlineLearningSettings = settings
        self.symbolics = symbolics

        # Initialize other variables
        self.x = symbolics()

        def proper_bounds(lower_bound, upper_bound):
            if not (lower_bound == -float('inf') and upper_bound == float('inf')):
                return True
            return False

        # Initialize dictionaries for optimization variables, online learning problems and their solvers
        self.observation_variables = {}
        self.observation_problem = {}
        self.observation_solver = {}
        self.bounds_dict = {}

        # Initialize blocks of variables
        self.block_indices = {}
        self.block_keys = {}

        # Handle quadratic penalty for g
        self.bool_g_quad = bool_g_quad
        if bool_g_quad:
            g_common = sum(g_dict.values(), [])
            shared_penalty_parameters = symbolics.sym('shared_penalty', len(g_common))
            penalty_parameters = cs.vertcat(penalty_parameters, shared_penalty_parameters)
            shared_reward += - 0.5 * cs.sum1(shared_penalty_parameters * cs.vertcat(*g_common) ** 2)
            g_dict = {}

        g = sum(g_dict.values(), [])
        self.nb_g = len(g)
        h = sum(h_dict.values(), [])
        self.nb_h = len(h)
        lbv = sum(lbv_dict.values(), [])
        ubv = sum(ubv_dict.values(), [])
        v = cs.vertcat(v_dict.values())
        self.nb_v = v.numel()

        f_player_g = lambda player: player.player_g
        f_player_h = lambda player: player.player_h
        player_constraints_g = list(chain.from_iterable((f_player_g(player)) for player in players.values()))
        player_constraints_h = list(chain.from_iterable((f_player_h(player)) for player in players.values()))

        opt = cs.vertcat(*[player.opt for player in players.values()])

        # Set-up the symbolics, Lagrangian, constraints etc for the online learning methodology
        # The cost function of the observer
        V_full = symbolics()
        # The parameter vector of the observer
        p_full = symbolics()
        # The penalty vector of the observer
        penalty_full = symbolics()
        # The vector containing the optimization variables (v denotes the 'additional' optimization variables)
        x_v_full = symbolics()
        # The full vector of equality constraints for the observer
        g_full = symbolics()
        # The full vector of inequality constraints for the observer
        h_full = symbolics()
        # The full vector of inequality constraints for the observer corresponding to parametric rectangle bounds
        h_full_bounds = symbolics()
        # The full vector of Lagrangian parameters mu for the observer
        mu_full = symbolics()
        # The full vector of mu times h for the observer
        mu_times_h_full = symbolics()
        # The full vector of mu times h for the observer
        mu_times_h_full_bounds = symbolics()
        # The full vector of Lagrangian parameters lambda for the observer
        lambda_full = symbolics()

        # Make lists for x0, x and the parameters
        x0_list = [p.x0 for p in players.values()]
        x_list = [p.opt for p in players.values()]
        parameters_list = [p_dict[key] for key in p_dict]
        nb_mu_bounds = 0

        bounds_original = make_bounds(players, g, h, lbv, ubv)

        # Iterate over the number of previous observations used to update the parameter
        for t in range(settings.nb_observations):
            # Make CasADi symbolics for optimization variables and parameters at this time stamp
            x0_step = symbolics(0,1)
            x_step = symbolics(0,1)
            v_step = symbolics(0,1)
            parameters_step = symbolics(0,1)
            p_step = symbolics(0,1)
            u_step = symbolics(0,1)
            lambda_step = symbolics.sym('lambda_step_' + str(t), len(g) + len(player_constraints_g), 1)
            mu_step = symbolics.sym('mu_step_' + str(t), len(h) + len(player_constraints_h), 1)

            # Initialize the box constraints
            def get_bound(lower, upper, var):
                try:
                    if lower == -float('inf'):
                        return upper - var
                except RuntimeError:
                    pass
                try:
                    if upper == float('inf'):
                        return var - lower
                except RuntimeError:
                    pass
                return (upper - var) * (var - lower)

            mu_step_bounds = symbolics.sym('', 0, 1)
            h_step_bounds = symbolics.sym('', 0, 1)

            # Define box constraints for optimizations variables
            for index, bounds in enumerate(zip(bounds_original['lbx'], bounds_original['ubx'])):
                lower_bound, upper_bound = bounds
                if proper_bounds(lower_bound, upper_bound):
                    mu_step_bounds = cs.vertcat(mu_step_bounds, symbolics.sym(
                        'mu_step_bounds_' + str(t), 1, 1))
                    bound = get_bound(lower_bound, upper_bound, opt[index])
                    h_step_bounds = cs.vertcat(h_step_bounds, bound)

            # Define box constraints for shared optimizations variables v
            for index, bounds in enumerate(zip(lbv, ubv)):
                lower_bound, upper_bound = bounds
                if proper_bounds(lower_bound, upper_bound):
                    mu_step_bounds = cs.vertcat(mu_step_bounds, symbolics.sym(
                        'mu_step_bounds_' + str(t), 1, 1))
                    bound = get_bound(lower_bound, upper_bound, v[index])
                    h_step_bounds = cs.vertcat(h_step_bounds, bound)

            # Set-up Lagrangian at the current time stamp
            cost = -sum([player.reward for player in players.values()]) - shared_reward
            if len(g) + len(player_constraints_g) > 0:
                cost -= cs.dot(lambda_step, cs.vertcat(*g, *player_constraints_g))
            if len(h) + len(player_constraints_h) > 0:
                cost -= cs.dot(mu_step, cs.vertcat(*h, *player_constraints_h))
            cost -= cs.sum1(mu_step_bounds * h_step_bounds)
            lagrangian = cs.jacobian(cost, cs.vertcat(opt, v))

            # Set-up x0 variables
            for i, p in players.items():
                sym = symbolics.sym('x0_' + str(i) + '_step_' + str(t), p.x0.shape)
                x0_step = cs.vertcat(x0_step, sym)
                p_step = cs.vertcat(p_step, sym)

            # Set-up x variables
            for i, p in players.items():
                sym_x = symbolics.sym('u_' + str(i) + '_step_' + str(t), p.opt.numel(), 1)
                sym_p = sym_x[:p.trajectory.dyn.nu,:]
                sym_u = sym_x[p.trajectory.dyn.nu:,:]
                x_step = cs.vertcat(x_step, sym_x)
                p_step = cs.vertcat(p_step, sym_p)
                u_step = cs.vertcat(u_step, sym_u)

            # Set-up parameters
            for i in p_dict:
                sym = symbolics.sym('x0_' + str(i) + '_step_' + str(t), p_dict[i].shape)
                parameters_step = cs.vertcat(parameters_step, sym)
                p_step = cs.vertcat(p_step, sym)

            # Set-up penalty parameters
            penalty_step = symbolics.sym('rho_step_' + str(t), penalty_parameters.shape)

            # Set-up additional optimization variables
            v_step = cs.vertcat(v_step, symbolics.sym('v_step_' + str(t), v.shape))

            # Substitute Lagrangian and constraints with new optimization variables
            V_step = cs.substitute(lagrangian,
                                cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
            g_step = cs.substitute(cs.vertcat(*g, *player_constraints_g),
                                cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
            h_step = cs.substitute(cs.vertcat(*h, *player_constraints_h),
                                cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))
            h_step_bounds = cs.substitute(h_step_bounds,
                                        cs.vertcat(*x0_list, *x_list, v, *parameters_list, penalty_parameters),
                                        cs.vertcat(x0_step, x_step, v_step, parameters_step, penalty_step))

            # Add new variables to the full vectors storing this information
            x_v_full = cs.vertcat(x_v_full, u_step, v_step)
            V_full = cs.horzcat(V_full, V_step)
            g_full = cs.vertcat(g_full, g_step)
            h_full = cs.vertcat(h_full, h_step)
            # h_full_bounds = cs.vertcat(h_full_bounds, h_step_bounds[self.parametric_bounds_index_dict[id]])
            p_full = cs.vertcat(p_full, p_step)
            penalty_full = cs.vertcat(penalty_full, penalty_step)
            mu_full = cs.vertcat(mu_full, mu_step, mu_step_bounds)
            nb_mu_bounds += mu_step_bounds.numel()
            if h_step.numel() > 0:
                mu_times_h_full = cs.vertcat(mu_times_h_full, mu_step * h_step)
            if h_step_bounds.numel() > 0:
                mu_times_h_full_bounds = cs.vertcat(mu_times_h_full_bounds, mu_step_bounds * h_step_bounds)
            lambda_full = cs.vertcat(lambda_full, lambda_step)

        # Add regularization term and handle reward parameters
        other_params = ct.struct_symSX([ct.entry(str(i), sym=player.reward_params) for i, player in players.items() if i != ego_id])

        self.two_norm_lagrangian = cs.sumsqr(V_full)

        nb_reward_params = other_params.size
        current_estimate_parameters = symbolics.sym('current_estimate_parameters', nb_reward_params, 1)
        try:
            regularization = symbolics.sym('reg', len(settings.regularization))
        except:
            regularization = symbolics.sym('reg')
        V_full = cs.sumsqr(V_full) + cs.sum1(regularization * (other_params - current_estimate_parameters) ** 2)

        # Set up parameter vector
        p_full = cs.vertcat(p_full, penalty_full, players[ego_id].reward_params, current_estimate_parameters, regularization)

        # Store all required information in a single dictionary
        self.observation_variables = {'theta': other_params, 'x_v': x_v_full, 'lambda': lambda_full,
                                        'mu': mu_full, 'f': V_full, 'p': p_full, 'g': g_full, 'h': h_full,
                                        'h_bounds': h_full_bounds, 'mu_times_h': mu_times_h_full,
                                        'mu_times_h_bounds': mu_times_h_full_bounds}

        self.observation_variables['x'] = cs.vertcat(self.observation_variables['theta'], self.observation_variables['x_v'],
                            self.observation_variables['lambda'], self.observation_variables['mu'])

        nb_lambdas = self.observation_variables['lambda'].numel()
        nb_mus = self.observation_variables['mu'].numel()

        self.bounds = make_full_bounds(players, g, h, lbv, ubv, nb_reward_params, nb_lambdas, nb_mus, nb_mu_bounds, settings.nb_observations)

        # Set up the optimization problem
        self.setup_solver(players)

    def reset(self):
        return

    def setup_solver(self, players):
        """ Builds the solver for the online learning methodology for each human player with unknown parameters """
        print("Build Learner")
        # for id, player in players.items():
        #     if id != self.id:
        keys = ['theta', 'x_v', 'lambda', 'mu']
        current_index = 0
        self.block_indices = {}
        self.block_keys = []
        for key in keys:
            sx = self.observation_variables[key]
            if sx.shape[0] > 0:
                new_index = current_index + sx.shape[0]
                self.block_indices[key] = (current_index, new_index)
                self.block_keys.append(key)
                current_index = new_index

        g = cs.vertcat(self.observation_variables['g'], self.observation_variables['mu_times_h'],
            self.observation_variables['mu_times_h_bounds'], self.observation_variables['h'])
        g1 = cs.vertcat(self.observation_variables['g'], self.observation_variables['h'])
        g2 = cs.vertcat(self.observation_variables['mu_times_h'], self.observation_variables['mu_times_h_bounds'])

        self.observation_problem = {'x': self.observation_variables['x'],
                                    'f': self.observation_variables['f'],
                                    'p': self.observation_variables['p'],
                                    'g': g, 'g1': g1, 'g2': g2}

        # set-up solver for online learning methodology
        if self.settings.solver == 'ipopt':
            self.observation_solver = src.solvers.get_ipopt_solver(self.observation_problem, self.settings, self.bounds)

        else:
            self.primal_factor = self.settings.panoc_delta_tolerance / self.settings.panoc_delta_tolerance_primal_feas
            self.complementarity_factor = self.settings.panoc_delta_tolerance / self.settings.panoc_delta_tolerance_complementarity
            self.complementarity_bounds_factor = self.settings.panoc_delta_tolerance / self.settings.panoc_delta_tolerance_complementarity_bounds
            factor = self.symbolics.sym('factor', 3)
            self.observation_problem['p'] = cs.vertcat(self.observation_variables['p'], factor)
            self.observation_problem['g'] = cs.vertcat(
                        factor[0]*self.observation_variables['g'],
                        factor[0]*self.observation_variables['mu_times_h'],
                        factor[1]*self.observation_variables['mu_times_h_bounds'],
                        factor[2]*self.observation_variables['h'])
            self.observation_problem['g1'] = cs.vertcat(
                        factor[0]*self.observation_variables['g'],
                        factor[2]*self.observation_variables['h'])
            self.observation_problem['g2'] = cs.vertcat(
                        factor[0]*self.observation_variables['mu_times_h'],
                        factor[1]*self.observation_variables['mu_times_h_bounds'])

            if self.settings.solver == 'OpEn':
                self.observation_solver = src.solvers.get_OpEn_solver(self.observation_problem, self.settings, self.bounds, self.id, None, "learn")
            else:
                self.observation_solver = src.solvers.get_panocpy_solver(self.observation_problem, self.settings, self.bounds, self.id, None, "learn")

    def observe(self, players, observations):
        """ Initializes the bounds for the players in the GPG

        Parameters
        ----------
        players : dict
            the current estimate for the parameters of the human players
        observations : deque object
            contains the required information for initializing and warm-starting the online learning methodology
        """
        if observations.maxlen == 0 or len(observations) < observations.maxlen:
            return
        # new_estimate_parameters_dict = {}

        # Set-up optimization variables and parameter lists in the correct form based on the given observation object
        p_numeric = []
        penalty_numeric = []
        x_v0_numeric = []
        mu_bounds_numeric = []
        mu_numeric = []
        lambda_numeric = []
        for observation in observations:
            p_step = []
            u_step = []
            v_step = [] # TODO: check v_step
            mu_bounds_numeric_step = []
            mu_numeric_step = []
            lambda_numeric_step = []
            if len(observation) == 7:
                for i, value in observation[0].items():
                    p_step = cs.vertcat(p_step, value)
                for i in observation[1].keys():
                    if i == 'common':
                        v_step = observation[1][i]
                    else: #TODO
                        p_step = cs.vertcat(p_step, observation[2][int(i)][:players[int(i)].trajectory.dyn.nu])
                        u_step = cs.vertcat(u_step, observation[1][i][players[int(i)].trajectory.dyn.nu:])
                        mu_bounds_numeric_step = cs.vertcat(mu_bounds_numeric_step, observation[3][i])
                        mu_numeric_step = cs.vertcat(mu_numeric_step, observation[4]['g_' + i])
                        lambda_numeric_step = cs.vertcat(lambda_numeric_step, observation[4]['h_' + i])
                for i, value in observation[5].items():
                    p_step = cs.vertcat(p_step, value)
                penalty_step = cs.vertcat(observation[6], observation[4]['g_common']) if self.bool_g_quad else observation[6]

                p_numeric = cs.vertcat(p_numeric, p_step)
                penalty_numeric = cs.vertcat(penalty_numeric, penalty_step)
                x_v0_numeric = cs.vertcat(x_v0_numeric, u_step, v_step)
                mu_bounds_numeric = cs.vertcat(mu_bounds_numeric, mu_bounds_numeric_step)
                mu_numeric = cs.vertcat(mu_numeric, mu_numeric_step) if self.bool_g_quad else cs.vertcat(mu_numeric, observation[4]['g_common'], mu_numeric_step)
                lambda_numeric = cs.vertcat(lambda_numeric, lambda_numeric_step, observation[4]['h_common'])
            else:
                raise Exception('Observation format is incorrect')
        if cs.vertcat(mu_numeric, mu_bounds_numeric).numel() == 0:
            nb_lambdas = self.observation_variables['lambda'].numel()
            nb_mus = self.observation_variables['mu'].numel()
            lambda_numeric = cs.DM.zeros(nb_lambdas, 1)
            mu_numeric = cs.DM.zeros(nb_mus, 1)
            mu_bounds_numeric = cs.DM.zeros(0, 1)


        reward_params_current_belief = cs.vertcat(*(player.reward_params_current_belief for i, player in players.items() if i != self.id))

        p_numeric = cs.vertcat(p_numeric, penalty_numeric, players[self.id].reward_params_current_belief, reward_params_current_belief, self.settings.regularization)
        x = cs.vertcat(reward_params_current_belief, x_v0_numeric, lambda_numeric, mu_numeric, mu_bounds_numeric)
        # Solve online learning optimization problem
        if self.settings.solver == 'ipopt':
            # solution = self.observation_solver_dict[id](x0=x,
            #                                             p=p_numeric,
            #                                             lbx=self.lbx_dict[id],
            #                                             ubx=self.ubx_dict[id],
            #                                             lbg=0,
            #                                             ubg=self.ubg_dict[id])
            solution, self.cpu_time = self.observation_solver(x, p_numeric)
            x = solution['x']
            a, b = self.block_indices['theta']
            reward_params_struct = self.observation_variables["theta"](list(x[a:b].toarray(True)))
            
        elif self.settings.solver == 'OpEn':
            solution = self.observation_solver(x.toarray(True),cs.vertcat(p_numeric, self.complementarity_factor,
                                                                                    self.complementarity_bounds_factor, self.primal_factor).toarray(True))
            self.cpu_time = solution.solve_time_ms / 1000
            x = solution.solution
            a, b = self.block_indices['theta']
            reward_params_struct = self.observation_variables["theta"](x[a:b])
        else:
            x, mu, stats = self.observation_solver(x.toarray(True),cs.vertcat(p_numeric, self.complementarity_factor,
                                                                                        self.complementarity_bounds_factor, self.primal_factor).toarray(True))
            self.cpu_time = stats['elapsed_time'].total_seconds()
            a, b = self.block_indices['theta']
            reward_params_struct = self.observation_variables["theta"](x[a:b])

        for i, player in players.items():
            if i != self.id:
                player.reward_params_current_belief = reward_params_struct[str(i)]

        print('Online learning:  2-norm of Lagrangian: ' + str(
            cs.substitute(self.two_norm_lagrangian,
                        cs.vertcat(self.observation_variables['x'], self.observation_variables['p']),
                        cs.vertcat(x, p_numeric))))
        print('New estimate parameters = ' + str(reward_params_struct.master))
        return