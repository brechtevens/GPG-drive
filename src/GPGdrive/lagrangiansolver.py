from casadi.casadi import solve
import numpy as np
import casadi as cs
from .penalty import PenaltyUpdater
from . import solvers
from .settings import GPGSolverSettings

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

    lbg_player = list(chain.from_iterable((f_lbg(player)) for player in players.values()))
    ubg_player = list(chain.from_iterable((f_ubg(player)) for player in players.values()))
    lbh_player = list(chain.from_iterable((f_lbh(player)) for player in players.values())) 
    ubh_player = list(chain.from_iterable((f_ubh(player)) for player in players.values()))
    bounds = {
        'lbx': list(chain.from_iterable((f_lb(player)) for player in players.values())),
        'ubx': list(chain.from_iterable((f_ub(player)) for player in players.values())),
        'lbg': lbg_common + lbg_player + lbh_player + lbh_common,
        'ubg': ubg_common + ubg_player + ubh_player + ubh_common,
        'lbg1': lbg_player + lbh_player + lbh_common,
        'ubg1': ubg_player + ubh_player + ubh_common,
        'lbg2': lbg_common,
        'ubg2': ubg_common,
    }
    bounds['lbx'].extend(lbv)
    bounds['ubx'].extend(ubv)
    return bounds


def vertcat(list):
    return cs.vertcat(*list) if bool(list) else cs.SX(0,1)

class solver(object):
    """
    A class used to solve GPG formulations using a single reformulation with an additional quadratic penalty method
    """

    def __init__(self, identifier, players, shared_reward, g_dict, h_dict, v_dict, lbv_dict, ubv_dict, p_dict, penalty_parameters, solver_settings:GPGSolverSettings):
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
        solver_settings : GPGSolverSettings object
            the solver settings for the GPG
        """
        assert (solver_settings.solver in ['ipopt', 'OpEn', 'panocpy'])
        self.players = players
        self.id = identifier
        self.solver_settings = solver_settings
        
        self.nb_inner_iterations = 0
        self.nb_outer_iterations = 0
        self.cost = 0
        self.stage_cost = 0
        self.constraint_violation = 0
        self.cpu_time = 0

        g = sum(g_dict.values(), [])
        self.nb_g = len(g)
        h = sum(h_dict.values(), [])
        lbv = sum(lbv_dict.values(), [])
        ubv = sum(ubv_dict.values(), [])
        v = vertcat(v_dict.values())

        self.bounds = make_bounds(players, g, h, lbv, ubv)
    
        self.problem = {
            # 'x': cs.vertcat(*[player.opt for player in players.values()], v),
            'x': ct.struct_symSX([*[ct.entry(str(i), sym=player.opt) for i, player in players.items()], ct.entry("common", sym=v)]),
            'f': -sum([player.reward for player in players.values()]) - shared_reward,
            'p':  cs.vertcat(*[player.x0 for player in players.values()],
                             *[player.reward_params for player in players.values()],
                             *[p_dict[key] for key in p_dict.keys()], penalty_parameters.all),
            # 'g': cs.vertcat(*g, *player_constraints_g, *player_constraints_h, *h)
            'g': ct.struct_SX([
                ct.entry('g_common', expr=vertcat(g)),
                *[ct.entry('g_' + str(i), expr=vertcat(player.player_g)) for i, player in players.items()],
                *[ct.entry('h_' + str(i), expr=vertcat(player.player_h)) for i, player in players.items()],
                ct.entry('h_common', expr=vertcat(h)),
                ]),
            'g1': ct.struct_SX([
                *[ct.entry('g_' + str(i), expr=vertcat(player.player_g)) for i, player in players.items()],
                *[ct.entry('h_' + str(i), expr=vertcat(player.player_h)) for i, player in players.items()],
                ct.entry('h_common', expr=vertcat(h)),
                ]),
            'g2': ct.struct_SX([
                ct.entry('g_common', expr=vertcat(g)),
                ]),
        }

        self.penalty_handler = PenaltyUpdater(penalty_parameters)
        self.penalty_handler.generate_constraint_function([self.problem['x'], self.problem['p']])

        self.reset()

        # Create function to evaluate the cost function
        self.cost_function = cs.Function('f_player', [self.problem['x'], self.problem['p']],
                                    [cs.substitute(-players[self.id].reward, penalty_parameters.all, cs.DM.zeros(penalty_parameters.nb_all,1))])

        self.f_prob = cs.Function('f', [self.problem['x'], self.problem['p']], [self.problem['f']])
        self.g_prob = cs.Function('g', [self.problem['x'], self.problem['p']], [self.problem['g']])

        if solver_settings.solver == 'ipopt':
            # Initialize solver for 'ipopt'
            self.solver = solvers.get_ipopt_solver(self.problem, self.solver_settings, self.bounds)

        elif solver_settings.solver == 'OpEn':
            OpEn_problem = self.problem.copy()
            OpEn_problem['x'] = self.problem['x'].cat
            OpEn_problem['g'] = self.problem['g'].cat
            OpEn_problem['g1'] = self.problem['g1'].cat
            OpEn_problem['g2'] = self.problem['g2'].cat
            self.solver = solvers.get_OpEn_solver(OpEn_problem, self.solver_settings, self.bounds, self.id)
        else:
            self.solver = solvers.get_panocpy_solver(self.problem, self.solver_settings, self.bounds, self.id)
        return

    def reset_x_v(self):
        self.x_numeric_struct = self.problem['x'](0)

    def reset(self):
        # Reset initial penalty parameter values
        self.penalty_handler.reset_penalty_parameters()

        # Reset x and v numeric
        self.init = True
        self.reset_x_v()

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
            for player_id, player in players.items():
                self.x_numeric_struct.cat[self.x_numeric_struct.f[str(player_id)][:-player.trajectory.dyn.nu]] = self.x_numeric_struct.cat[self.x_numeric_struct.f[str(player_id)][player.trajectory.dyn.nu:]]
                self.x_numeric_struct.cat[self.x_numeric_struct.f[str(player_id)][-player.trajectory.dyn.nu:]] = 0

            # Shift state and control variables
            # for i, player in players.items():
            #     nu = player.trajectory.dyn.nu
            #     nx = player.trajectory.dyn.nx
            #     self.x_numeric_dict[i][0:player.nb_u - nu] = self.x_numeric_dict[i][nu:player.nb_u]
            #     self.x_numeric_dict[i][player.nb_u - nu:player.nb_u] = [0] * nu

            #     if player.mode == 'multiple':
            #         self.x_numeric_dict[i][player.nb_u:player.nb_opt - nx] = self.x_numeric_dict[i][player.nb_u + nx: player.nb_opt]
            #         next_state = player.trajectory.dyn(self.x_numeric_dict[i][player.nb_opt - nx:player.nb_opt], self.x_numeric_dict[i][player.nb_u - nu:player.nb_u])
            #         next_state = player.vehicle.substitute_reward_params(next_state)
            #         self.x_numeric_dict[i][player.nb_opt - nx:player.nb_opt] = next_state
        
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

        theta_list = cs.vertcat(*[player.reward_params_current_belief for player in players.values()])

        if self.penalty_handler.nb_all > 0:
            while self.nb_outer_iterations == 0 or \
                    not self.penalty_handler.terminated and self.nb_outer_iterations < self.solver_settings.panoc_max_outer_iterations:
                self.minimize_problem(players, x0_numeric_dict, theta_list, p_numeric_dict)
                self.nb_outer_iterations += 1
        else:
            self.nb_outer_iterations = self.minimize_problem(players, x0_numeric_dict, theta_list, p_numeric_dict)

        print('GNEP:        ID: ' + str(self.id) + ',   constraint violation: ' + str(self.constraint_violation))
        return self.x_numeric_struct, self.mu_bounds_numeric_struct, self.mu_numeric_struct, self.penalty_handler.values

    def minimize_problem(self, players, x0_numeric_dict, theta_list, p_numeric_dict):
        """ Minimizes the optimization problem of a single player

        Parameters
        ----------
        id : int
            the identifier of the player
        x0_numeric_dict : dict
            the numeric values of the initial states of the players of the GPG
        theta_list : list
            the numeric values of the current estimates of the parameters of the human players in the GPG
        p_numeric_dict : dict
            the numeric values for the parameters of the optimization problems
        """
        # Solve the optimization problem
        x_old = self.x_numeric_struct.cat

        params = cs.vertcat(*[x0_numeric_dict[key] for key in players],
                                 theta_list, *[p_numeric_dict[key] for key in p_numeric_dict], self.penalty_handler.values).toarray(True)
        if self.solver_settings.solver == 'ipopt':
            solution, self.cpu_time = self.solver(x_old, params)

            self.constraint_violation = 0
            x_opt = solution['x'].toarray(True)
            lam_x_opt = solution['lam_x'].toarray(True)
            mu_opt = solution['lam_g'].toarray(True)
            outer_iterations = 0
            
        elif self.solver_settings.solver == 'OpEn':
            solution = self.solver(x_old.toarray(True), params)

            if solution.exit_status != 'Converged':
                print(solution.exit_status)
                print(solution.num_outer_iterations)
                print(solution.num_inner_iterations)

            x_opt = solution.solution

            self.cpu_time += solution.solve_time_ms / 1000
            self.nb_inner_iterations += solution.num_inner_iterations
            self.constraint_violation = solution.f2_norm
            outer_iterations = solution.num_outer_iterations

            lam_x_opt = 0
            constraints_num = self.g_prob(x_opt, params)
            mu_opt = solution.penalty * cs.vertcat(cs.fmin(0.0, constraints_num[:self.nb_g]), constraints_num[self.nb_g:])
        else:
            x_opt, mu_opt, stats = self.solver(x_old, params)
            lam_x_opt = 0

            if stats['status'].value != 1:
                print(stats['status'])
                
            self.cpu_time += stats['elapsed_time'].total_seconds()
            self.nb_inner_iterations += stats['inner']['iterations']

            if 'δ' in stats:
                self.constraint_violation = stats['δ']
            else:
                self.constraint_violation = max(stats['δ₁'], stats['δ₂'])
            outer_iterations = stats['outer_iterations']


        self.mu_bounds_numeric_struct = self.problem['x'](lam_x_opt)
        self.mu_numeric_struct = self.problem['g'](mu_opt)
        self.x_numeric_struct = self.problem["x"](x_opt)

        self.cost = self.cost_function(x_opt, params)

        # Update the penalty handler
        self.constraint_violation = cs.fmax(self.penalty_handler.constraint_violation([x_opt, params]), self.constraint_violation)
        self.penalty_handler.update_index_set([x_opt, params])
        self.penalty_handler.update_penalty_parameters()
        return outer_iterations


