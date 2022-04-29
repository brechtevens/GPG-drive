import casadi as cs
import os
import numpy as np
from . import collision as collision
from . import car as car
import pickle


class Logger:
    """
    A class used to represent a highway with multiple lanes

    Attributes
    ----------
    """
    def __init__(self, experiment):
        self.settings = experiment.logger_settings
        self.save_on_draw = False
        self.generate_video = False
        self.history = {}
        self.history_keys = ['x', 'y', 'angle', 'velocity', 'acceleration', 'steering angle',
                             'stage cost', 'cost', 'effective constraint violation', 'planned constraint violation',
                             'number of Gauss-Seidel iterations', 'number of penalty iterations',
                             'GPG_wall_time', 'learning_wall_time', 'GPG_cpu_time', 'learning_cpu_time', 'belief']
        self.squared_distances = None
        self.Ts = experiment.world.Ts
        self.reset(experiment.world.cars)
        self.pickle_experiment_setup(experiment)

    @property
    def location(self):
        experiments_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'experiments', self.settings.name_experiment)
        return experiments_dir

    def pickle_experiment_setup(self, experiment):
        self.init_logger_folders()
        with open(self.location + 'experiment', "wb") as f:
            pickle.dump(experiment, f)
        return

    def init_logger_folders(self):
        """ Initializes the folders for saving the experiment if they do not exist yet """
        time_path = self.location + '/time'
        iter_path = self.location + '/iter'
        additional_path = self.location + '/analyze'
        video_path = self.location + '/video'
        if not os.path.exists(self.location):
            os.makedirs(self.location)
            os.makedirs(time_path)
            os.makedirs(iter_path)
            os.makedirs(additional_path)
            if self.settings.save_video:
                os.makedirs(video_path)
        else:
            if not os.path.exists(self.location + '/video'):
                os.makedirs(video_path)
            else:
                for file in os.listdir(video_path):
                    if file.endswith('.mp4'):
                        os.remove(video_path + "/" + file)

    def reset(self, cars):
        for key in self.history_keys:
            self.history[key] = {}
            for vehicle in cars:
                if key in self.history_keys[:6]:
                    self.history[key][vehicle.id] = []
                elif key in self.history_keys[6:16]:
                    if isinstance(vehicle, car.GPGOptimizerCar):
                        self.history[key][vehicle.id] = []
                else:
                    if isinstance(vehicle, car.GPGOptimizerCar):
                        self.history[key][vehicle.id] = {}
                        for player_id in vehicle.players:
                            if player_id != vehicle.id:
                                self.history[key][vehicle.id][player_id] = []
        if len(cars) == 2:
            self.history['headway'] = []
            self.squared_distances = collision.pointwise_projection_formulation_inequality_constraints(cars[0], cars[1])
        self.history['potential'] = []

    def log_data(self, cars):
        # Add data to history dict for visualization and optionally saving
        self.history['potential'].append(0)
        x_dict = {}
        for vehicle in cars:
            x_dict[vehicle.id] = vehicle.x
        for vehicle in cars:
            self.history['x'][vehicle.id].append(vehicle.x[0])
            self.history['y'][vehicle.id].append(vehicle.x[1])
            self.history['angle'][vehicle.id].append(vehicle.x[2])
            self.history['velocity'][vehicle.id].append(vehicle.x[3])
            self.history['acceleration'][vehicle.id].append(vehicle.u_2D[0])
            self.history['steering angle'][vehicle.id].append(vehicle.u_2D[1])
            if isinstance(vehicle, car.GPGOptimizerCar):
                r = vehicle.get_current_reward(vehicle.x, vehicle.u, x_dict, 0)
                violation = 0
                g_list = vehicle.stage_g + vehicle.soft_stage_g + vehicle.ego.player_stage_g
                h_list = vehicle.stage_h + vehicle.soft_stage_h + vehicle.ego.player_stage_h
                for con in g_list:
                    violation = cs.fmax(violation, abs(min(con(x_dict))))
                for con in h_list:
                    if not isinstance(cs.fmin(0.0, min(con(x_dict))), cs.SX):
                        violation = cs.fmax(violation, abs(cs.fmin(0.0, min(con(x_dict)))))
                self.history['effective constraint violation'][vehicle.id].append(violation)
                self.history['planned constraint violation'][vehicle.id].append(vehicle.optimizer.constraint_violation)
                self.history['potential'][-1] -= r
                self.history['stage cost'][vehicle.id].append(-r)
                self.history['cost'][vehicle.id].append(vehicle.optimizer.cost)
                self.history['number of Gauss-Seidel iterations'][vehicle.id].append(vehicle.optimizer.nb_inner_iterations)
                self.history['number of penalty iterations'][vehicle.id].append(vehicle.optimizer.nb_outer_iterations)
                self.history['GPG_wall_time'][vehicle.id].append(vehicle.gpg_solution_time)
                self.history['learning_wall_time'][vehicle.id].append(vehicle.observer_solution_time)
                self.history['GPG_cpu_time'][vehicle.id].append(vehicle.optimizer.cpu_time)
                self.history['learning_cpu_time'][vehicle.id].append(vehicle.observer.cpu_time)

                for player_id, player in vehicle.players.items():
                    if player_id != vehicle.id:
                        self.history['belief'][vehicle.id][player_id].append(player.reward_params_current_belief)

        if len(cars) == 2:
            x = {0: cars[0].x, 1: cars[1].x}
            self.history['headway'].append(cs.sqrt(min(self.squared_distances(x))))

    def write_data_to_files(self, current_iteration):
        exit_status = 0
        if self.settings.nb_iterations_experiment is not None:
            if self.settings.save_video:
                self.save_on_draw = True
            if current_iteration == self.settings.nb_iterations_experiment:
                # Stop experiment and save data
                exit_status = 1
                self.save_experiment()
                if self.settings.save_video:
                    self.generate_video = True
        return exit_status

    def make_statistics(self, path, key, i):
        dtype = [('Min', np.float64), ('Max', np.float64), ('Mean', np.float64), ('Std', np.float64)]
        nb_skipped = sum(elem < 1e-5 for elem in self.history[key][i])
        if nb_skipped != len(self.history[key][i]):
            structured_array = np.array([(np.min(self.history[key][i][nb_skipped:]),
                                          np.max(self.history[key][i][nb_skipped:]),
                                          np.mean(self.history[key][i][nb_skipped:]),
                                          np.std(self.history[key][i][nb_skipped:]))], dtype=dtype)
            np.savetxt(path + '/' + key + '_statistics_' + str(i) + '.csv',
                       structured_array, delimiter=',', fmt='%20.15f', header='Min,Max,Mean,Std', comments='')
        return

    def save_experiment(self):
        self.init_logger_folders()
        """ Saves the experiment data """
        time_path = self.location + '/time'
        iter_path = self.location + '/iter'
        additional_path = self.location + '/analyze'
        if not self.settings.only_save_statistics:
            for key, data in self.history.items():
                if key == 'headway' or key == 'potential':
                    iter_data = np.vstack(
                        (range(1, self.settings.nb_iterations_experiment + 1), cs.vertcat(*data).toarray(True))).transpose()
                    time_data = np.vstack(
                        ([i * self.Ts for i in range(self.settings.nb_iterations_experiment)], cs.vertcat(*data).toarray(True))) \
                        .transpose()
                    np.savetxt(iter_path + '/' + str(key) + '.dat', np.nan_to_num(iter_data), delimiter='\t', fmt='%16.10f')
                    np.savetxt(time_path + '/' + str(key) + '.dat', np.nan_to_num(time_data), delimiter='\t', fmt='%16.10f')
                else:
                    for id, dataline in data.items():
                        if key == 'belief':
                            for id_other, datalineline in dataline.items():
                                datalineline = cs.horzcat(*datalineline)
                                for i in range(datalineline.shape[0]):
                                    belief_list = datalineline[i, :].toarray(True)
                                    iter_data = np.vstack((range(1,
                                                                 self.settings.nb_iterations_experiment + 1),
                                                           belief_list)).transpose()
                                    time_data = np.vstack(([i * self.Ts for i in range(
                                        self.settings.nb_iterations_experiment)],
                                                           belief_list)).transpose()
                                    np.savetxt(iter_path + '/' + str(key) + '_' + str(id) + '_' + str(
                                        id_other) + '_' + str(i) + '.dat', iter_data, delimiter='\t',
                                               fmt='%16.10f')
                                    np.savetxt(time_path + '/' + str(key) + '_' + str(id) + '_' + str(
                                        id_other) + '_' + str(i) + '.dat', time_data, delimiter='\t',
                                               fmt='%16.10f')
                        else:
                            iter_data = np.vstack((range(1, self.settings.nb_iterations_experiment + 1), cs.vertcat(*dataline).toarray(True))).transpose()
                            time_data = np.vstack(([i * self.Ts for i in
                                                    range(self.settings.nb_iterations_experiment)],
                                                    cs.vertcat(*dataline).toarray(True))).transpose()
                            np.savetxt(iter_path + '/' + str(key) + '_' + str(id) + '.dat', iter_data,
                                       delimiter='\t', fmt='%16.10f')
                            np.savetxt(time_path + '/' + str(key) + '_' + str(id) + '.dat', time_data,
                                       delimiter='\t', fmt='%16.10f')

            for id in self.history['x']:
                np.savetxt(additional_path + '/trajectory_' + str(id) + '.dat',
                           np.vstack((cs.vertcat(*self.history['x'][id]).toarray(True), cs.vertcat(*self.history['y'][id]).toarray(True))).transpose(),
                           delimiter='\t', fmt='%16.10f')

            for id in self.history['stage cost']:
                self.make_statistics(additional_path, 'GPG_wall_time', id)
                self.make_statistics(additional_path, 'GPG_cpu_time', id)
                self.make_statistics(additional_path, 'learning_wall_time', id)
                self.make_statistics(additional_path, 'learning_cpu_time', id)
                dtype = [('Closed loop cost', np.float64)]
                structured_array = np.array([(np.sum(self.history['stage cost'][id]))], dtype=dtype)
                np.savetxt(additional_path + '/closed_loop_cost_' + str(id) + '.csv',
                           structured_array, delimiter=',', fmt='%20.15f', header='Closed loop cost', comments='')
                dtype = [('Effective constraint violation', np.float64)]
                structured_array = np.array([(np.max(self.history['effective constraint violation'][id]))],
                                            dtype=dtype)
                np.savetxt(additional_path + '/effective_constraint_violation_' + str(id) + '.csv',
                           structured_array, delimiter=',', fmt='%20.15f',
                           header='Maximum effective constraint violation', comments='')
                dtype = [('Planned constraint violation', np.float64)]
                structured_array = np.array([(np.max(self.history['planned constraint violation'][id]))],
                                            dtype=dtype)
                np.savetxt(additional_path + '/planned_constraint_violation_' + str(id) + '.csv',
                           structured_array, delimiter=',', fmt='%20.15f',
                           header='Maximum planned constraint violation', comments='')
            dtype = [('Closed loop potential', np.float64)]
            structured_array = np.array([(np.sum(self.history['potential']))], dtype=dtype)
            np.savetxt(additional_path + '/closed_loop_potential.csv',
                       structured_array, delimiter=',', fmt='%20.15f', header='Closed loop potential', comments='')
        else:
            assert (self.settings.statistics_index is not None)
            for id in self.history['Gauss-Seidel solution time']:
                self.make_statistics(additional_path, 'GPG_wall_time', id)
                self.make_statistics(additional_path, 'GPG_cpu_time', id)
                self.make_statistics(additional_path, 'learning_wall_time', id)
                self.make_statistics(additional_path, 'learning_cpu_time', id)
        return
