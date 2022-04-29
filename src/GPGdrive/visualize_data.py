import matplotlib.pyplot as plt
import casadi as cs


def plot(data_dict, windows_dict, nb_iterations=None):
    """ Plots the demanded windows of the given dataset

    Parameters
    ----------
    data_dict : dict
        a dictionary containing the data of the experiment
    windows_dict : dict
        a dictionary containing the windows (str) and the corresponding list of relevant identifiers
    nb_iterations : int
        the number of iterations which should be visualized
    """
    colors = ['red', 'yellow', 'blue', 'gray', 'orange', 'purple', 'white']
    if nb_iterations is None:
        nb_iterations = len(data_dict['x'][0])

    if nb_iterations > 0:
        for window, id_list in windows_dict.items():
            plt.figure()
            if window == 'headway':
                plt.plot(range(nb_iterations), cs.DM(cs.vertcat(*data_dict['headway'][:nb_iterations])))
            if window == 'potential':
                plt.plot(range(nb_iterations), cs.DM(cs.vertcat(*data_dict[window][:nb_iterations])))
            elif window == 'belief':
                belief_data = cs.transpose(cs.horzcat(*data_dict['belief'][id_list[0]][id_list[1]]))
                for i in range(belief_data.shape[1]):
                    plt.plot(range(nb_iterations), belief_data[:, i], color=colors[id_list[0]])
            else:
                for i in id_list:
                    plt.plot(range(nb_iterations), cs.DM(cs.vertcat(*data_dict[window][i][:nb_iterations])), color=colors[i])
            plt.xlabel('iteration')
            plt.ylabel(window)
            # plt.legend(loc='best')
        plt.show()
    return
