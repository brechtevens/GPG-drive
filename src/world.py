import src.feature as feature
import numpy as np
import src.scene as scene
import casadi as cs
import src.collision as collision
import src.car as car

class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)


class World(object):
    """
    A class used to represent traffic scenarios

    Attributes
    ----------
    cars : list
        the cars in the traffic scenario
    lanes : list
        the lanes of the traffic scenario
    roads : list
        the roads of the traffic scenario
    highway : Highway object
        the highway of the traffic scenario, if any
    _colors : list
        the possible colors for the cars in the scenario
    Ts : float
        the sampling time of the experiment, required to save the timestamps of the current experiment correctly
    """
    def __init__(self):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.scene = scene.Scene()
        self.crossroad = None
        self._colors = ['red', 'yellow', 'blue', 'white', 'orange', 'purple', 'gray']
        self.Ts = 0

    def set_racetrack(self):
        import environments.racetrack as racetrack
        self.scene = racetrack.RaceTrack()

    def set_nb_lanes(self, nb_lanes, width=3.0, length_list=None):
        """ Sets the number of lanes for a traffic scenario on a highway section

        Parameters
        ----------
        nb_lanes : int
            the number of lanes on the highway section (> 0)
        width : float
            the width of the lanes
        length_list : list
            the lengths of the different lanes
         """
        assert (nb_lanes > 0)
        self.scene = scene.Highway([-1., 0.], [0., 0.], width, nb_lanes, length_list)
        self.lanes = self.scene.get_lanes()
        self.roads.append(self.scene)

    def make_crossroads(self, nb_lanes, width=3.0):
        """ Sets the number of lanes for a traffic scenario on a highway section

        Parameters
        ----------
        nb_lanes : int
            the number of lanes on each side of the road
        width : float
            the width of the lanes
        length_list : list
            the lengths of the different lanes
         """
        assert (nb_lanes > 0)
        self.scene = scene.Crossroad([0., 0.], [1, 0.], width, nb_lanes)
        self.lanes = self.scene.get_lanes()
        self.roads.append(self.scene)

    def add_vehicle(self, type, dynamics, x0, horizon=None, color=None, **vargs):
        """ Adds a vehicle to the traffic scenario and returns its identifier

        Parameters
        ----------
        type : str
            the vehicle type, eg 'UserControlledCar'
        dynamics : Dynamics object
            the dynamics of the vehicle
        x0 : list
            the initial state of the vehicle
        horizon : int, optional
            the control horizon of the vehicle
        color : str, optional
            the color of the vehicle
         """
        id = len(self.cars)
        if horizon is None:
            horizon = 1
        if color is None:
            color = self._colors[id]
        vehicle_initializer = getattr(car, type)
        self.cars.append(vehicle_initializer(dynamics, x0, horizon, id, color, **vargs))
        return id

    def thesis_reward(self, C_v_des, v_des, C_road=0., additional_reward=None, dynamics=None):
        """ Returns a commonly used stage reward, a parameterized version and the cost function parameters

        consists of control feature, velocity feature and optionally a feature for driving at the center of the road, i.e.
            reward = C_v_des * (v - v_des)^2 + C_road * highway.quadratic() + feature.control()

        Parameters
        ----------
        C_v_des : float
            parameter value for keeping the desired velocity
        v_des : float
            the desired velocity
        C_road : float, optional
            parameter value for driving at the center of the road
         """
        if dynamics is not None and hasattr(dynamics, "parametric_bounds"):
            control_feature = feature.control(dynamics.parametric_bounds)
        else:
            control_feature = feature.control()
        if C_road == 0:
            params = cs.SX.sym('theta_human', 2, 1)
            reward = control_feature + C_v_des * feature.speed(v_des)
            reward_parametrized = control_feature + params[0] * feature.speed(params[1])
        else:
            params = cs.SX.sym('theta_human', 3, 1)
            reward = control_feature + C_v_des * feature.speed(v_des) - C_road * self.scene.quadratic()
            reward_parametrized = control_feature + params[0] * feature.speed(params[1]) - params[2] * self.scene.quadratic()
        if additional_reward is not None:
            reward += additional_reward
            reward_parametrized += additional_reward
        return reward, reward_parametrized, params

    def set_reward(self, id, reward, terminal_reward=None, params=None, param_values=None, shared_reward=None):
        """ Sets the stage reward and terminal reward of a given vehicle in the traffic scenario

        Parameters
        ----------
        id : int
            the identifier of the vehicle
        reward : Feature
            the stage reward of the vehicle
        terminal_reward : Feature
            the terminal reward of the vehicle
         """
        if isinstance(self.cars[id], car.GPGOptimizerCar):
            self.cars[id].add_player(self.cars[id], reward, terminal_reward, params, param_values)
        else:
            print('Could not add reward!')
        if shared_reward is not None:
            self.cars[id].stage_shared_reward = shared_reward
        return

    def add_human(self, id_player, id_human, human_reward, human_terminal_reward=None, params=None, param_values=None, avoid_collisions=False):
        """ Adds a 'human' vehicle to a GPGOptimizerCar

        Parameters
        ----------
        id_player : int
            the id of the GPGOptimizerCar
        id_human : int
            the id of the human
        human_reward : Feature
            the reward of the human
        human_terminal_reward : Feature
            the terminal reward of the human
        params : cs.SX or cs.MX
            the parameters of the human subproblem
        param_values : list or float
            the initial estimate for the human parameters
         """
        if isinstance(self.cars[id_player], car.GPGOptimizerCar):
            self.cars[id_player].add_player(self.cars[id_human], human_reward, human_terminal_reward, params, param_values)
            if avoid_collisions:
                pass #TODO
        return

    def add_obstacle(self, id_player, id_obstacle):
        """ Adds a 'obstacle' to a GPGOptimizerCar

        Parameters
        ----------
        id_player : int
            the id of the GPGOptimizerCar
        id_obstacle : int
            the id of the obstacle
        """
        if isinstance(self.cars[id_player], car.GPGOptimizerCar):
            self.cars[id_player].add_obstacle(self.cars[id_obstacle])
        return

    def set_collision_avoidance_mode(self, mode, *args):
        """ Sets the collision avoidance mode for all vehicles

        Parameters
        ----------
        mode : str
            the string for the collision avoidance formulation, i.e. pointwise_projection, product or dual
        """
        if mode == 'projection':
            self.add_common_constraints(collision.projection_formulation_inequality_constraints, 'add_h', *args)
        elif mode == 'ellipse':
            self.add_common_constraints(collision.ellipse_formulation_inequality_constraints, 'add_h', *args)
        elif mode == 'pointwise_projection':
            self.add_common_constraints(collision.pointwise_projection_formulation_inequality_constraints, 'add_h', *args)
        elif mode == 'product':
            self.add_common_constraints(collision.product_formulation_equality_constraints, 'add_g', *args)
        elif mode == 'product_simplified':
            self.add_common_constraints(collision.product_formulation_equality_constraints_simplified, 'add_g', *args)
        elif mode == 'dual':
            self.add_common_constraints(collision.dual_formulation_constraints, 'add_dual', *args)
        else:
            raise Exception('The given collision avoidance mode is unknown')

    def add_common_constraints(self, constraint_formulation, method, *args):
        """ Adds the common constraints for all vehicles

        Parameters
        ----------
        constraint formulation : Constraints object
            the common constraints
        method : str
            determines whether equality or inequality constraints are added, equals 'add_h' or 'add_g'
        """
        for id in range(len(self.cars)):
            self.set_common_constraints(id, constraint_formulation, method, *args)
        return

    def set_common_constraints(self, id, constraint_formulation, method, *args):
        """ Sets the common constraints of a single vehicle

        Parameters
        ----------
        id : int
            the identifier of the regarded vehicle
        constraint formulation : Constraints object
            the common constraints
        method : str
            determines whether equality or inequality constraints are added, equals 'add_h' or 'add_g'
        epsilon : float, optional
            the epsilon parameter for the collision avoidance constraints, i.e. the virtual enlargement
        """
        method_to_call = getattr(car.GPGOptimizerCar, method)
        vehicle = self.cars[id]
        if isinstance(vehicle, car.GPGOptimizerCar):
            for i, player in vehicle.players.items():
                if i != id:
                    other_vehicle = player.vehicle
                    method_to_call(vehicle, constraint_formulation(vehicle, other_vehicle, *args))
                    print('added constraint between ' + str(vehicle.id) + ' and ' + str(other_vehicle.id) + ' for ' + str(vehicle.id))
                    if isinstance(other_vehicle, car.GPGOptimizerCar):
                        for j, other_other_player in other_vehicle.players.items():
                            if j != other_vehicle.id:
                                other_other_vehicle = other_other_player.vehicle
                                if other_other_vehicle.id != id:
                                    method_to_call(vehicle, constraint_formulation(other_vehicle, other_other_vehicle, *args))
                                    print('added constraint between ' + str(other_vehicle.id) + ' and ' + str(other_other_vehicle.id) + ' for ' + str(vehicle.id))
                        for j, other_other_obstacle in other_vehicle.obstacles.items():
                            other_other_vehicle = other_other_obstacle.vehicle
                            method_to_call(vehicle, constraint_formulation(other_vehicle, other_other_vehicle, *args))
                            print('added constraint between ' + str(other_vehicle.id) + ' and ' + str(other_other_vehicle.id) + ' for ' + str(vehicle.id))
            for i, other_obstacle in vehicle.obstacles.items():
                other_vehicle = other_obstacle.vehicle
                method_to_call(vehicle, constraint_formulation(vehicle, other_vehicle, *args))
                print('added constraint between ' + str(vehicle.id) + ' and ' + str(other_vehicle.id) + ' for ' + str(vehicle.id))
        return

    def add_boundary_constraint(self, id, *args):
        """ Adds a boundary constraint for a single given vehicle

        Parameters
        ----------
        id : int
            the identifier of the regarded vehicle
        """
        for vehicle in self.cars:
            if isinstance(vehicle, car.GPGOptimizerCar):
                if id in vehicle.players.keys():
                    if self.scene.boundary_constraint(self.cars[id]).type == "inequality":
                        vehicle.add_player_h(id, self.scene.boundary_constraint(self.cars[id], *args))
                    else:
                        vehicle.add_player_g(id, self.scene.boundary_constraint(self.cars[id], *args))
        return
