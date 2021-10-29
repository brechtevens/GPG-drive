from src.settings import GPGSolverSettings
import casadi as cs


class PenaltyParameters(object):

    def __init__(self, sym, N, settings):
        self._sym = sym
        self._stage_penalty_parameters = sym()
        self._terminal_penalty_parameters = sym()
        self._stage_violation = sym()
        self._terminal_violation = sym()
        self._N = N
        self._settings:GPGSolverSettings = settings
        self.nb_z = 0

    @property
    def settings(self):
        return self._settings

    @property
    def stage(self):
        return self._stage_penalty_parameters

    @property
    def terminal(self):
        return self._terminal_penalty_parameters

    @property
    def all(self):
        return cs.vertcat(cs.reshape(self._stage_penalty_parameters, self._stage_penalty_parameters.numel(), 1),
                          self._terminal_penalty_parameters)

    @property
    def all_violation(self):
        return cs.vertcat(cs.reshape(self._stage_violation, self._stage_violation.numel(), 1),
                          self._terminal_violation)

    @property
    def nb_stage(self):
        return self._stage_penalty_parameters.numel()

    @property
    def nb_terminal(self):
        return self._terminal_penalty_parameters.numel()

    @property
    def nb_all(self):
        return self._stage_penalty_parameters.numel() + self._terminal_penalty_parameters.numel()

    @property
    def N(self):
        return self._N

    def get_violation(self, constraint, type):
        if type == 'stage_g':
            return cs.vertcat(*constraint)
        else:
            return cs.fmin(0, cs.vertcat(*constraint))

    def _add_stage(self, stage_constraint, constraint_type):
        assert (len(stage_constraint) % self._N == 0)
        stage_constraint_length = len(stage_constraint) // self._N

        # Add new penalty parameters
        rho_new = self._sym.sym('penalty', stage_constraint_length, self._N)
        self._stage_penalty_parameters = cs.vertcat(self._stage_penalty_parameters, rho_new)

        # Add violations
        violation = self.get_violation(stage_constraint, constraint_type)
        self._stage_violation = cs.vertcat(self._stage_violation, cs.reshape(violation, stage_constraint_length, self._N))
        return rho_new

    def _add_terminal(self, terminal_constraint, constraint_type):
        rho_new = self._sym.sym('penalty', len(terminal_constraint), 1)

        # Add new penalty parameters
        self._terminal_penalty_parameters = cs.vertcat(self._terminal_penalty_parameters, rho_new)

        # Add violations
        self._terminal_violation = cs.vertcat(self._terminal_violation,
                                             self.get_violation(terminal_constraint, constraint_type))
        return rho_new

    def add_penalty_constraints(self, current_constraint, constraint_type):
        """ Add penalty constraints with given penalty type

        Parameters
        ----------
        constraint_type : str
            the type of the constraints
        current_constraint : Sized
            the soft, penalty constraints
        """

        if constraint_type == 'stage_g' or constraint_type == 'stage_h':
            rho = self._add_stage(current_constraint, constraint_type)
        elif constraint_type == 'terminal_h':
            rho = self._add_terminal(current_constraint, constraint_type)
        else:
            raise Exception('The given constraint type is unknown')

        if self._settings.penalty_method == 'quadratic':
            reward_contribution = -cs.sum1(cs.reshape(rho/2, len(current_constraint), 1) *
                                           self.get_violation(current_constraint, constraint_type) ** 2)
            z = None
        elif self._settings.penalty_method == 'exact_naive' or\
                (constraint_type == 'stage_g' and self._settings.penalty_method == 'exact'):
            reward_contribution = -cs.sum1(cs.reshape(rho, len(current_constraint), 1) *
                                           self.get_violation(current_constraint, constraint_type))
            z = None
        elif self._settings.penalty_method == 'exact':
            z = self._sym.sym('z_h', len(current_constraint), 1)
            self.nb_z += len(current_constraint)
            reward_contribution = -cs.sum1(cs.reshape(rho, len(current_constraint), 1) * z)
        else:
            raise Exception('The given penalty method is unknown')
        return reward_contribution, z


class PenaltyUpdater(object):

    def __init__(self, penalty_parameters):
        self._penalty_parameters:PenaltyParameters = penalty_parameters
        self._settings:GPGSolverSettings = penalty_parameters.settings
        self._penalty_parameter_values = cs.DM.ones(self._penalty_parameters.nb_all) *\
                                         self._settings.panoc_initial_penalty
        self._constraint_function = None
        self._penalty_update_function = None
        self._indices_to_increase = []
        self._max_penalty_value = self._settings.panoc_initial_penalty *\
                                  self._settings.panoc_penalty_weight_update_factor ** self._settings.panoc_max_outer_iterations

    @property
    def nb_all(self):
        return self._penalty_parameters.nb_all
    
    @property
    def terminated(self):
        return len(self._indices_to_increase) == 0

    @property
    def values(self):
        return self._penalty_parameter_values

    @property
    def index_set(self):
        return self._indices_to_increase

    def reset_penalty_parameters(self):
        self._penalty_parameter_values = cs.DM.ones(self._penalty_parameters.nb_all) *\
                                         self._settings.panoc_initial_penalty

    def update_index_set(self, p_current):
        if self._constraint_function is None:
            self._indices_to_increase = []
        else:
            # if self._settings.penalty_norm == 'norm_inf':
            violation_vector = cs.fabs(self._constraint_function(*p_current)) > self._settings.panoc_delta_tolerance
            self._indices_to_increase = [i for i in range(violation_vector.numel()) if violation_vector[i]]
            # elif self._settings.penalty_norm == 'norm_2':
            #     self._indices_to_increase = cs.norm_inf(self._constraint_function(p_current)) > self._settings.panoc_delta_tolerance
        return

    def generate_constraint_function(self, constraint_variables):
        if self._penalty_parameters.nb_all > 0:
            self._constraint_function = cs.Function('constraint_test', constraint_variables,
                            [self._penalty_parameters.all_violation])
        else:
            self._constraint_function = None

    def constraint_violation(self, p_current):
        if self._constraint_function is None:
            return 0
        else:
            # if self._settings.penalty_norm == 'norm_inf':
            return cs.mmax(cs.fabs(self._constraint_function(*p_current)))
            # elif self._settings.penalty_norm == 'norm_2':
                # return cs.norm_2(self._constraint_function(*p_current))
            return

    def update_penalty_parameters(self):
        if self._settings.penalty_update_rule == 'individual':
            self._penalty_parameter_values[self._indices_to_increase] *= self._settings.panoc_penalty_weight_update_factor
            self._penalty_parameter_values = cs.fmin(self._penalty_parameter_values, self._max_penalty_value)
        if self._settings.penalty_update_rule == 'timed':
            raise NotImplementedError
        if self._settings.penalty_update_rule == 'collective':
            self._penalty_parameter_values *= self._settings.panoc_penalty_weight_update_factor
        return

    def shift_penalty_parameters(self):
        N = self._penalty_parameters.N
        M = self._penalty_parameters.nb_stage//N
        for k in range(N-1):
            self._penalty_parameter_values[k*M:(k+1)*M] = cs.fmax(self._settings.panoc_initial_penalty, self._penalty_parameter_values[(k+1)*M:(k+2)*M]/self._settings.panoc_penalty_weight_update_factor**3)
        self._penalty_parameter_values[(N-1) * M:N * M] = cs.DM.ones(M) * self._settings.panoc_initial_penalty
        # self._penalty_parameter_values[:N * M] = cs.fmax(self._settings.panoc_initial_penalty,
        #                             self._penalty_parameter_values[:N * M]/(self._settings.panoc_penalty_weight_update_factor**2))
        self._penalty_parameter_values[N * M:] = cs.DM.ones(self._penalty_parameters.nb_terminal) *\
                                                 self._settings.panoc_initial_penalty

