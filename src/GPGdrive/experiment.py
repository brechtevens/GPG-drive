#!/usr/bin/env python
from . import settings as settings
from . import world as world


class Experiment(object):
    """
    A class used to represent experiments

    Attributes
    ----------
    name_experiment : str
        the name of the world of the experiment
    experiment_variant : str, optional
        the variant of the experiment
    pyglet_visualization_settings : PygletVisualizationSettings object
        the visualization settings for the pyglet window
    data_visualization_windows : dict
        the data visualization windows, shown as pop-up matplotlib plots
    logger_settings: LoggerSettings object
        the settings for the data logger of the experiment
    """
    def __init__(self, name_experiment, experiment_variant=None):
        # Initialize variables
        self.name_experiment = name_experiment
        self.experiment_variant = experiment_variant
        self.world = None
        self.data_visualization_windows = None

        # Initialize default settings
        if experiment_variant is None:
            directory_name = name_experiment
        else:
            directory_name = name_experiment
            for i, param in enumerate(experiment_variant):
                if i == 0:
                    directory_name += '/' + param
                else:
                    directory_name += '-' + param
        self.solver_settings = settings.GPGSolverSettings(name_experiment)
        self.learning_settings = settings.OnlineLearningSettings(name_experiment)
        self.logger_settings = settings.LoggerSettings(directory_name)
        self.pyglet_visualization_settings = settings.PygletVisualizationSettings()

    # def build_world(self):
    #     try:
    #         self.world = getattr(world, 'world_' + self.name_experiment)(self.solver_settings, self.learning_settings,
    #                                                                      *self.experiment_variant[1:])
    #         self.data_visualization_windows = settings.default_data_visualization_windows(self.world)
    #     except:
    #         pass
    #     return

    def set_data_visualization_window(self, window, id_list):
        """ Adds a visualization for the current run

        See plot_trajectories.py for more info

        Parameters
        ----------
        window : str
            name of the visualized variable
        id_list : list
            list of ids relevant for visualizing the variable
        """
        self.data_visualization_windows[window] = id_list

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['world']
        del odict['data_visualization_windows']
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
        self.build_world()
