import numpy as np
from . import feature as feature 
import casadi as cs


class Trajectory(object):
    """
    A class used to represent general trajectory objects for dynamical systems

    Attributes
    ----------
    dyn : Dynamics object
        the dynamics of the trajectory
    N : int
        the length of the trajectory
    x0: list
        the current state of the trajectory
    u: list
        the control sequence along the trajectory
    """
    def __init__(self, N, dyn):
        self.dyn = dyn
        self.N = N
        self.x0 = [0] * dyn.nx
        self.u = [0] * (dyn.nu * N)

    def get_future_trajectory(self):
        """ Returns the future states of the trajectory for the current control sequence """
        trajectory = []
        z = self.x0
        for k in range(self.N):
            z = self.dyn(z, self.u[k * self.dyn.nu:(k + 1) * self.dyn.nu])
            trajectory.append(z)
        return trajectory

    def get_future_trajectory_given_u(self, u):
        """ Returns the future states of the trajectory for the given control sequence

        Parameters
        ----------
        u : CasADi SX or MX
            the given control sequence
        """
        trajectory = []
        z = self.x0
        for k in range(u.numel()//self.dyn.nu):
            z = self.dyn(z, u[k * self.dyn.nu:(k + 1) * self.dyn.nu])
            trajectory.append(z)
        return trajectory

    def quadratic_following_reward(self, d_des, target_vehicle):
        """ Returns a cost feature rewarding driving at a certain distance from a preceding vehicle

        Parameters
        ----------
        d_des : float
            the desired headway distance
        target_vehicle : Car object
            the preceding target vehicle
        """
        lr_front = target_vehicle.dyn.lr
        lf_back = self.dyn.lf
        headway = feature.headway(False, lr_front, lf_back)

        @feature.feature
        def f(x, u, x_other, k):
            return -(d_des - headway(x, u, x_other[target_vehicle.id], k)) ** 2
        return f

    def shared_reward(self, stage_shared_reward, players):
        """ Returns the total shared reward along the entire trajectory

        Parameters
        ----------
        stage_shared_reward : Feature
            the stage shared reward
        players : dict
            all physical objects
        """
        shared_reward = 0
        for k in range(self.N):
            x_k = {}
            for i, player in players.items():
                x_k[i] = player.x[k]
            shared_reward += stage_shared_reward(None, None, x_k, k)
        return shared_reward

    def reward(self, stage_reward, x_robot, u_robot, others, terminal_reward=None):
        """ Returns the total reward along the entire trajectory for an ego vehicle

        Parameters
        ----------
        stage_reward : Feature
            the stage reward of the ego vehicle
        x_robot : list
            the state variables of the ego vehicle along the trajectory
        u_robot : list
            the control variables of the ego vehicle along the trajectory
        others : dict
            the surrounding physical objects required to evaluate the reward along the trajectory
        terminal_reward : Feature, optional
            the terminal reward
        """
        reward = 0
        for k in range(self.N):
            x_k_others = {}
            for i, other in others.items():
                x_k_others[i] = other.x[k]
            reward += stage_reward(x_robot[k], u_robot[k * self.dyn.nu:(k + 1) * self.dyn.nu], x_k_others, k)
            if terminal_reward is not None and k == self.N-1:
                reward += terminal_reward(x_robot[k], u_robot[k * self.dyn.nu:(k + 1) * self.dyn.nu], x_k_others)
        return reward

    def constraints(self, stage_constraints, others):
        """ Returns the constraints along the entire trajectory for an ego vehicle

        Parameters
        ----------
        stage_constraints : Constraints
            the stage constraints of the ego vehicle
        others : dict
            the surrounding physical objects required to evaluate the constraints along the trajectory
        """
        if stage_constraints.length == 0:
            return np.array([])
        constraint_list = []
        for k in range(self.N):
            x_k = {}
            u_k = {}
            for player_id in stage_constraints.id_list:
                x_k[player_id] = others[player_id].x[k]
                u_k[player_id] = others[player_id].u[k]
            constraint_list.extend(stage_constraints(x_k, u_k))
        return constraint_list

    def constraints_dual_formulation(self, dual_formulation, others, symbolics):
        """ Returns the dual constraints along the entire trajectory for an ego vehicle

        Parameters
        ----------
        dual_formulation : Constraints
            the stage dual constraints of the ego vehicle
        others : dict
            the surrounding physical objects required to evaluate the constraints along the trajectory
        symbolics: cs.SX or cs.MX attribute
            allows to generate additional SX or MX symbolics
        """
        lam = symbolics.sym('dual_lam', 4, 1)
        mu = symbolics.sym('dual_mu', 4, 1)
        x_0 = {}
        for i in dual_formulation.id_list:
            x_0[i] = others[i].x[0]
        if len(dual_formulation(x_0, lam, mu)) == 0:
            return np.array([]), np.array([]), symbolics()
        inequality_constraint_list = []
        equality_constraint_list = []
        optimization_parameters = symbolics()
        for k in range(self.N):
            lam = symbolics.sym('dual_lam_' + str(k), 4, 1)
            mu = symbolics.sym('dual_mu_' + str(k), 4, 1)
            x_k = {}
            for i in dual_formulation.id_list:
                x_k[i] = others[i].x[k]
            g, h = dual_formulation(x_k, lam, mu)
            equality_constraint_list.extend(g)
            inequality_constraint_list.extend(h)
            optimization_parameters = cs.vertcat(optimization_parameters, lam, mu)
        return equality_constraint_list, inequality_constraint_list, optimization_parameters
