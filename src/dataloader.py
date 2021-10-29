##########################################################################################     IMPORT SECTION

import numpy as np


##########################################################################################     FUNCTIONS

def matrixToFile(mat, filepath):
    """
        Function that writes given a numpy-matrix mat
        to a file at the specified path 'filepath'.
    """
    with open(filepath, 'w') as f:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                # NOTE: Python stores numbers correctly to about 16 or 17 digits.
                # 8 digits should largely suffice in our practical context.

                # Last delimiter must be omitted in order to load in matrix easier in Python.
                if j != (len(mat[0])-1):
                    f.write("{:.8f}, ".format(mat[i][j]))
                else:
                    f.write("{:.8f}".format(mat[i][j]))
            f.write("\n")
    f.close()
    return


def loadMatrix(filepath, delim=', ', dtype=None):
    """
        Function which returns a matrix that is stored in file at given path 'filepath'.
        delim is an optional argument that tells us what the delimiter is in the file. Default = ', '.
        dtype is an optional argument that tells us what the type of the elements of the matrix are that we load in.
    """
    mat = np.loadtxt(filepath, delimiter=delim, dtype=dtype)
    mat = mat[:, :len(mat[0])-1]        # In Matlab stored with last value = ' ', so we slice it out.
    if dtype != None:
        mat = mat.astype(np.float)      # np.loadtxt loads in matrix as np.array of dtype, so convert to floats.
    return mat
