import numpy as np
import casadi as cs
import opengen as og
from .penalty import PenaltyUpdater
import sys
import os.path
from . import solvers
from .settings import GPGSolverSettings

from itertools import chain

class solver(object):
    """
    A class used to solve GPG formulations using a Gauss-Seidel algorithm with an additional quadratic penalty method
    """

    def __init__(self, identifier, players, shared_reward, g_dict, h_dict, v_dict, lbv_dict, ubv_dict, p_dict, penalty_parameters, symbolics, solver_settings:GPGSolverSettings):
        """
        Parameters
        ----------
        identifier : int
            the identifier of the vehicle
        players : dict
            dictionary consisting of the players in the GPG
        shared_reward
            shared reward for all players
        g_dict : dict
            the shared equality constraints
        h_dict : dict
            the shared inequality constraints
        v_dict : dict
            the additional shared optimization variables
        lbv_dict : dict
            the lower bounds on the additional shared optimization variables
        ubv_dict : dict
            the upper bounds on the additional shared optimization variables
        p_dict : dict
            the parameters of the obstacles in the GPG or any additional parameters
        penalty_parameters : [cs.SX, cs.SX] or [cs.MX, cs.MX]
            the penalty parameters of the GPG
        symbolics: cs.SX or cs.MX attribute
            allows to generate additional SX or MX symbolics
        solver_settings : GPGSolverSettings object
            the solver settings for the GPG
        """
        assert (solver_settings.solver in ['ipopt', 'OpEn', 'panocpy'])
        self.init = True
        self.solver_settings = solver_settings
        self.id = identifier
        self.nb_inner_iterations = 0
        self.nb_outer_iterations = 0
        self.cost = 0
        self.stage_cost = 0
        self.constraint_violation = 0
        self.x_numeric_dict = {}
        self.player_mu_bounds_numeric = {}
        self.player_mu_numeric = {}
        self.player_lambda_numeric = {}

        g = tuple(*g_dict.values())
        self.nb_g = len(g)
        h = tuple(*h_dict.values())
        self.nb_h = len(h)
        self.lbv = list(*lbv_dict.values())
        self.ubv = list(*ubv_dict.values())
        v = cs.vertcat(v_dict.values())
        self.nb_v = v.numel()

        self.lbg_common = [0] * len(g)
        self.ubg_common = [0] * len(g)
        self.lbh_common = [0] * len(h)
        self.ubh_common = [float("inf")] * len(h)

        self.problem_dict = {}
        self.bounds_dict = {}
        self.solver_dict = {}
        self.cpu_time = 0

        self.total_nb_opt = 0
        self.player_h_index_dict = {}
        self.player_g_index_dict = {}
        
        self.x_numeric_dict = {}
        self.x_index_dict = {}
        for key, player in players.items():
            self.x_index_dict[key] = np.s_[self.total_nb_opt: self.total_nb_opt + player.nb_opt]
            self.total_nb_opt += player.nb_opt
            
            self.player_g_index_dict[key] = np.s_[self.nb_g: self.nb_g + player.nb_player_g]
            self.player_h_index_dict[key] = np.s_[self.nb_g + player.nb_player_g: self.nb_g + player.nb_player_g + player.nb_player_h]

        def get_all_parameters(players):
            return [cs.vertcat(*[player.x0 for player in players.values()],
                            *[player.opt for player in players.values()],
                            *[player.reward_params for player in players.values()],
                            *[p_dict[key] for key in p_dict.keys()], v)]

        all_parameters = get_all_parameters(players)

        self.penalty_handler = PenaltyUpdater(penalty_parameters)
        self.penalty_handler.generate_constraint_function(all_parameters)
        self.player_specific_constraint_violation = cs.Function('constraint_test', all_parameters,
                                                                [cs.vertcat(*[constraint for player in players.values() for constraint in player.player_g],
                                                                            *[cs.fmin(0.0, constraint) for player in players.values() for constraint in player.player_h])])

        # Create function to evaluate the cost function of each player
        self.cost_function = cs.Function('f', all_parameters,
                                         [cs.substitute(-players[self.id].reward, penalty_parameters.all, cs.DM.zeros(penalty_parameters.nb_all,1))])

        self.v_numeric = [0] * self.nb_v
        for id, player in players.items():
            self.x_numeric_dict[id] = [0] * player.nb_opt

        for i, player in players.items():
            x = cs.vertcat(player.opt, v)
            x_old = symbolics.sym('x_old_' + str(i), x.size())

            problem_parameters = cs.vertcat(*[p.x0 for p in players.values()],
                            *[p.opt for key, p in players.items() if key != i],
                            *[p.reward_params for p in players.values()],
                            *[p_dict[key] for key in p_dict.keys()], x_old, penalty_parameters.all)

            self.problem_dict[i] = {
                'x': x,
                'f': - shared_reward - player.reward + solver_settings.gs_regularization * cs.sumsqr(x - x_old),
                'p': problem_parameters,
                'g': cs.vertcat(*g, *player.player_g, *h, *player.player_h)
            }

            self.bounds_dict[i] = {
                'lbx': player.lb + self.lbv,
                'ubx': player.ub + self.ubv,
                'lbg': self.lbg_common + player.lbg,
                'ubg': self.ubg_common + player.ubg,
                'lbh': player.lbh + self.lbh_common,
                'ubh': player.ubh + self.ubh_common
            }

        # Initialize solvers
        if self.solver_settings.solver == 'ipopt':
            for i, player in players.items():
                self.solver_dict[i] = solvers.get_ipopt_solver(self.problem_dict[i], self.solver_settings, self.bounds_dict[i])
                                                  
        elif self.solver_settings.solver == 'OpEn':
            for i, player in players.items():
                self.solver_dict[i] = solvers.get_OpEn_solver(self.problem_dict[i], self.solver_settings, self.bounds_dict[i], self.id, i)
            
        else:
            for i, player in players.items():
                self.solver_dict[i] = solvers.get_panocpy_solver(self.problem_dict[i], self.solver_settings, self.bounds_dict[i], self.id, i)

    def player_specific_constraint_violation_norm(self, p_current):
        # if self.solver_settings.penalty_norm == 'norm_inf':
        return cs.mmax(cs.fabs(self.player_specific_constraint_violation(p_current)))
        # elif self.solver_settings.penalty_norm == 'norm_2':
            # return cs.norm_2(self.player_specific_constraint_violation(p_current))

    def reset(self):
        # Reset initial penalty parameter values
        self.penalty_handler.reset_penalty_parameters()

        # Reset x and v numeric
        self.init = True
        self.v_numeric = [0] * self.nb_v
        for key in self.x_numeric_dict:
            self.x_numeric_dict[key] = [0] * len(self.x_numeric_dict[key])

        # if self.solver == 'ipopt' or self.solver == 'qpoases':
        #     self.v_numeric = cs.DM.zeros(self.nb_v)
        #     for i, player in players:
        #         self.x_numeric_dict[i] = cs.DM.zeros(player.nb_opt)
        #         self.lam_x_numeric_dict[i] = cs.DM.zeros(player.nb_opt + self.nb_v)
        #         self.lam_g_numeric_dict[i] = cs.DM.zeros(len(self.ubg_dict[i]))
        # else:
        #     self.v_numeric = [0] * self.nb_v
        #     for i in self.id_list:
        #         self.x_numeric_dict[i] = [0] * self.nb_x_dict[i]

    def shift_vector_control_inputs(self, players, x0_numeric_dict):
        """ Shift the optimization variables by one time step

        Parameters
        ----------
        x0_numeric_dict : dict
            the numeric values of the new initial states for the players of the GPG
        """
        # Test whether this is the very first iteration of the GPG
        if self.init:
            # Initialize state and control variables
            self.init = False
            # for i, player in players.items():
            #     if player.nb_opt > player.nb_u:
            #         nu = self.dyn_dict[i].nu
            #         nx = self.dyn_dict[i].nx
            #         self.x_numeric_dict[i][player.nb_u:player.nb_u + nx] = cs.DM(
            #             self.dyn_dict[i](x0_numeric_dict[i], [0] * nu))
            #         for j in range(1, player.nb_u // nu):
            #             self.x_numeric_dict[i][player.nb_u + j * nx:player.nb_u + (j + 1) * nx] = cs.DM(
            #                 self.dyn_dict[i](
            #                     self.x_numeric_dict[i][player.nb_u + (j - 1) * nx:player.nb_u + j * nx],
            #                     [0] * nu))
        else:
            # Shift state and control variables
            for i, player in players.items():
                nu = player.trajectory.dyn.nu
                nx = player.trajectory.dyn.nx
                if player.mode == "multiple":
                    self.x_numeric_dict[i][0:player.nb_u - nu] =\
                        self.x_numeric_dict[i][nu:player.nb_u]
                self.x_numeric_dict[i][player.nb_u - nu:player.nb_u] = [0] * nu
                # if player.nb_opt > player.nb_u:
                #     if player.nb_opt - player.nb_u > nx:
                #         self.x_numeric_dict[i][player.nb_u:player.nb_opt - nx] = self.x_numeric_dict[i][
                #                                                                            player.nb_u + nx:
                #                                                                            player.nb_opt]
                #     self.x_numeric_dict[i][player.nb_opt - nx:player.nb_opt] = cs.DM(
                #         self.dyn_dict[i](self.x_numeric_dict[i][player.nb_opt - nx:player.nb_opt],
                #                          self.x_numeric_dict[i][player.nb_u - nu:player.nb_u]))

        # Warm starting of penalty parameters
        if self.solver_settings.warm_start:
            self.penalty_handler.shift_penalty_parameters()
        else:
            self.penalty_handler.reset_penalty_parameters()
        return

    def minimize(self, x0_numeric_dict, players, p_numeric_dict):
        """ Solves the GPG and returns a generalized Nash equilibrium

        Parameters
        ----------
        x0_numeric_dict : dict
            the numeric values of the new initial states for the players of the GPG
        players : dict
            dictionary consisting of the players in the GPG
        p_numeric_dict : dict
            the numeric values for the parameters of the optimization problems
        """
        # Initialize parameters, i.e. overhead costs
        self.cpu_time = 0
        self.shift_vector_control_inputs(players, x0_numeric_dict)
        self.nb_inner_iterations = 0
        self.nb_outer_iterations = 0
        delta_players = {self.id: float('inf')}

        theta_list = cs.vertcat(*[player.reward_params_current_belief for player in players.values()])

        p_current = cs.vertcat(*[x0_numeric_dict[key] for key in players],
                    *[self.x_numeric_dict[key] for key in players],
                    *theta_list,
                    *[p_numeric_dict[key] for key in p_numeric_dict],
                    *[self.v_numeric])

        # Main loop
        while self.nb_outer_iterations == 0 or \
                not self.penalty_handler.terminated and self.nb_outer_iterations < self.solver_settings.panoc_max_outer_iterations:
            self.penalty_handler.update_penalty_parameters()
            self.nb_inner_iterations = 0
            while self.nb_inner_iterations < self.solver_settings.gs_max_iterations and \
                    max(delta_players.values()) > self.solver_settings.gs_tolerance:
                for i in players:
                    delta_players[i] = self.minimize_player(i, x0_numeric_dict, players, theta_list, p_numeric_dict)
                self.nb_inner_iterations += 1
            p_current = cs.vertcat(*[x0_numeric_dict[key] for key in players],
                                   *[self.x_numeric_dict[key] for key in players],
                                   *theta_list,
                                   *[p_numeric_dict[key] for key in p_numeric_dict],
                                   *[self.v_numeric])
            self.nb_outer_iterations += 1
            # print('constraint violation: ' + str(self.penalty_handler.constraint_violation(p_current)))
            self.penalty_handler.update_index_set(p_current)
            # print('index_set: ' + str(self.penalty_handler.index_set))
            # print('penalty_parameters: ' + str(self.penalty_handler.values))
            delta_players = {self.id: float('inf')}

        # Evaluate the cost and the constraint violation of the obtained Nash equilibrium
        self.cost = self.cost_function(cs.vertcat(p_current))
        self.constraint_violation = cs.fmax(self.penalty_handler.constraint_violation(p_current),
                                            self.player_specific_constraint_violation_norm(p_current))
        print('GNEP:        ID: ' + str(self.id) + ',   nb_inner_iterations: ' + str(
            self.nb_inner_iterations) + ',   nb_outer_iterations: ' + str(
            self.nb_outer_iterations) + ',   constraint violation: ' + str(self.constraint_violation))
        return self.x_numeric_dict, self.v_numeric, self.player_mu_bounds_numeric, self.player_mu_numeric, self.player_lambda_numeric,\
            self.penalty_handler.values

    def minimize_player(self, id, x0_numeric_dict, players, theta_list, p_numeric_dict):
        """ Minimizes the optimization problem of a single player

        Parameters
        ----------
        id : int
            the identifier of the player
        x0_numeric_dict : dict
            the numeric values of the initial states of the players of the GPG
        players : dict
            dictionary consisting of the players in the GPG
        theta_list : list
            the numeric values of the current estimates of the parameters of the human players in the GPG
        p_numeric_dict : dict
            the numeric values for the parameters of the optimization problems
        """
        # Solve the optimization problem
        x_old = np.concatenate([self.x_numeric_dict[id], self.v_numeric])
        params = np.concatenate([*[x0_numeric_dict[key] for key in players], *[self.x_numeric_dict[key] for key in players if key != id],
                                 theta_list, *[p_numeric_dict[key] for key in p_numeric_dict], x_old, self.penalty_handler.values.toarray(True)])
        if self.solver_settings.solver == 'ipopt':
            solution, self.cpu_time = self.solver_dict[id](x_old, params)

            x_opt = solution['x'].toarray(True)
            lam_x_opt = solution['lam_x'].toarray(True)
            lam_g_opt = solution['lam_g'].toarray(True)

            self.player_mu_bounds_numeric[id] = lam_x_opt[:players[id].nb_opt]
            self.player_mu_numeric[id] = lam_g_opt[self.player_g_index_dict[id]]
            self.player_lambda_numeric[id] = lam_g_opt[self.player_h_index_dict[id]]

        elif self.solver_settings.solver == 'OpEn':
            solution = self.solver_dict[id](x_old, params)

            if solution.exit_status != 'Converged':
                print(solution.exit_status)
                print(solution.num_outer_iterations)
                print(solution.num_inner_iterations)

            self.cpu_time = solution.solve_time_ms / 1000
            x_opt = solution.solution

            lam_x_opt = [0] * len(x_opt)
            lam_g_opt = solution.lagrange_multipliers

            self.player_mu_bounds_numeric[id] = lam_x_opt[:players[id].nb_opt]
            # self.player_mu_numeric[id] = lam_g_opt[self.player_g_index_dict[id]]
            # self.player_lambda_numeric[id] = lam_g_opt[self.player_h_index_dict[id]]

        else:
            x_opt, mu_opt, stats = self.solver_dict[id](x_old, params)

            self.cpu_time += stats['elapsed_time'].total_seconds()
            self.nb_inner_iterations += stats['inner']['iterations']
            self.constraint_violation = stats['δ']

            self.player_mu_bounds_numeric = {key: [0] * player.nb_opt for key, player in players.items()}
            self.player_mu_numeric[id] = mu_opt[self.player_g_index_dict[id]]
            self.player_lambda_numeric[id] = mu_opt[self.player_h_index_dict[id]]

        # Evaluate the difference between the new and the previous solution
        delta = cs.norm_inf(x_old - x_opt)

        # Set the numeric dicts based on the obtained solution
        self.x_numeric_dict[id] = x_opt[:players[id].nb_opt]
        if self.nb_v > 0:
            self.v_numeric = x_opt[players[id].nb_opt:]
        return delta
