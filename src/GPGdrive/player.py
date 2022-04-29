from sys import gettrace
from .trajectory import Trajectory
import casadi as cs

class PhysicalObject(object):
    def __init__(self, N, vehicle, sym):
        self.vehicle = vehicle
        self.trajectory = Trajectory(N, vehicle.dyn)
        self._sym = sym
        return

    def setup_x0_and_u(self):
        self.x0 = self._sym.sym('x0_id' + str(self.vehicle.id), self.trajectory.dyn.nx)
        self.u = self._sym.sym('u_id' + str(self.vehicle.id), self.trajectory.dyn.nu * self.trajectory.N)
    
    def setup_x(self, mode):
        self.mode = mode
        if mode == "single":
            self.x = self.x_single_shooting(self.trajectory.dyn, self.x0, self.u)
            self.g_dyn = []
        else:
            self.x = self.x_multiple_shooting(self.trajectory.dyn)
            self.setup_dynamics_constraints_multiple_shooting()

    def x_single_shooting(self, object_dyn, x0, u):
        """ Returns a list of the state variables along a trajectory given initial state and control variables

        Parameters
        ----------
        object_dyn : Dynamics object
            the dynamics of the regarded object
        x0 : cs.SX or cs.MX
            the initial state
        u : cs.SX or cs.MX
            the control variables
        """
        x = []
        z = x0
        for k in range(self.trajectory.N):
            z = object_dyn(z, u[k * object_dyn.nu:(k + 1) * object_dyn.nu])
            x.append(z)
        return x

    def x_multiple_shooting(self, object_dyn):
        """ Returns a list of the state variables along a trajectory

        Parameters
        ----------
        object_dyn : Dynamics object
            the dynamics of the regarded object
        i : int
            the identifier of the object
        """
        x = []
        for k in range(self.trajectory.N):
            z = self._sym.sym('x_id' + str(self.vehicle.id) + '_t' + str(k + 1), object_dyn.nx)
            x.append(z)
        return x

    def dynamics_constraints_multiple_shooting(self, object_dyn, x0, u, x):
        """ Returns a list of CasADi expressions corresponding the the dynamics of the object, i.e. equality constraints

        Parameters
        ----------
        object_dyn : Dynamics object
            the dynamics of the regarded object
        x0 : cs.SX or cs.MX
            the initial state
        u : cs.SX or cs.MX
            the control variables
        x : cs.SX or cs.MX
            the state variables along the trajectory
        """
        dynamics_constraints = [x[0][i] - object_dyn(x0, u[0:object_dyn.nu])[i] for i in range(object_dyn.nx)]
        for k in range(1, self.trajectory.N):
            dynamics_constraints += [x[k][i] - object_dyn(x[k - 1], u[k * object_dyn.nu:(k + 1) * object_dyn.nu])[i]
                                     for i in range(object_dyn.nx)]
        return dynamics_constraints

    def setup_dynamics_constraints_multiple_shooting(self):
        """ Initializes the dynamics constraints for the players in the GPG (only for multiple shooting) """
        self.g_dyn = self.dynamics_constraints_multiple_shooting(self.trajectory.dyn, self.x0, self.u, self.x)
        return
    
class Player(PhysicalObject):
    def __init__(self, N, vehicle, sym, reward, terminal_reward=None, params=None, param_values=None):
        super().__init__(N, vehicle, sym)

        # PLAYER-SPECIFIC CONSTRAINTS
        # Equality constraints
        self.player_g = []
        self.player_stage_g = []

        # Inequality constraints
        self.player_h = []
        self.player_stage_h = []
        self.player_terminal_h = []

        # REWARDS
        self._stage_reward = reward
        self._terminal_reward = terminal_reward

        # PARAMETERS
        if params is not None and param_values is not None:
            self._reward_params = params
            self._reward_params_initial_belief = cs.vertcat(*param_values)
            self._reward_params_current_belief = cs.vertcat(*param_values)
        else:
            self._reward_params = self._sym()
            self._reward_params_initial_belief = []
            self._reward_params_current_belief = []
        return 

    @property
    def opt(self):
        if self.mode == "single":
            return self.u
        else:
            return cs.vertcat(self.u, cs.vertcat(*self.x))

    @property
    def nb_u(self):
        return self.u.numel()
    
    @property
    def nb_opt(self):
        return self.opt.numel()

    @property
    def nb_reward_params(self):
        return self._reward_params.numel()

    @property
    def reward_params(self):
        return self._reward_params

    @property
    def reward_params_initial_belief(self):
        return self._reward_params_initial_belief

    @property
    def reward_params_current_belief(self):
        return self._reward_params_current_belief

    @reward_params_current_belief.setter
    def reward_params_current_belief(self, value):
        self._reward_params_current_belief = value

    def reset_belief(self):
        if hasattr(self, "_reward_params"):
            self._reward_params_current_belief = self._reward_params_initial_belief

    def setup_rewards(self, others):
        self._reward = self.trajectory.reward(self._stage_reward, self.x, self.u, others, terminal_reward=self._terminal_reward)
        self._reward_original = self._reward

    @property
    def reward(self):
        return self._reward

    def setup_stage_constraints(self, others):
        self.player_g.extend(self.g_dyn)        
        for con in self.player_stage_g:
            self.player_g.extend(self.trajectory.constraints(con, others))
            
        for con in self.player_stage_h:
            self.player_h.extend(self.trajectory.constraints(con, others))
        
    def setup_terminal_constraints(self, x_N):
        for con in self.player_terminal_h:
            self.player_h.extend(con(x_N))
    
    @property
    def nb_player_g(self):
        return len(self.player_g)

    @property
    def nb_player_h(self):
        return len(self.player_h)

    @property
    def nb_player_constraints(self):
        return self.nb_player_g + self.nb_player_h

    @property
    def lbg(self):
        return [0] * self.nb_player_g

    @property
    def ubg(self):
        return [0] * self.nb_player_g

    @property
    def lbh(self):
        return [0] * self.nb_player_h

    @property
    def ubh(self):
        return [float("inf")] * self.nb_player_h

    def setup_bounds(self):
        if self.mode == 'single':
            self.lb = self.trajectory.dyn.bounds[0] * self.trajectory.N
            self.ub = self.trajectory.dyn.bounds[1] * self.trajectory.N
        else:
            self.lb = self.trajectory.dyn.bounds[0] * self.trajectory.N + [-float("inf")] * self.trajectory.dyn.nx * self.trajectory.N
            self.ub = self.trajectory.dyn.bounds[1] * self.trajectory.N + [float("inf")] * self.trajectory.dyn.nx * self.trajectory.N
        return

class Obstacle(PhysicalObject):
    def __init__(self, N, vehicle, sym):
        super().__init__(N, vehicle, sym)

        self.trajectory.x0 = vehicle.x
        self.trajectory.u = vehicle.fixed_control * N
