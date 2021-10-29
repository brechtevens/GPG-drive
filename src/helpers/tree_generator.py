import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import math

def generate_trees(nb_trees, width, height, radius=5, valid_check=None):

    # Generate 100 points (3-tuples) between 0 and 10
    x_points = np.random.uniform(-width, width,[nb_trees,1])
    y_points = np.random.uniform(-height, height,[nb_trees,1])

    points = np.hstack([x_points, y_points])

    # Pairwise distances between points
    distances = euclidean_distances(points)

    # "Remove" distance to itself by setting to a distance of radius+1 (to discard it later)
    distances += np.identity(len(distances)) * (radius+1)

    # Retrieve the distance to the closest point
    min_dist = np.min(distances,axis=1)

    # Filter your set of points
    filtered_points = points[min_dist>radius]

    # Apply valid checker
    if valid_check is not None:
        filtered_points = filtered_points[valid_check(filtered_points)]

    return filtered_points

def generate_trees_Gabor(nb_trees, width, height, radius=5, valid_check=None):

    def gentree():
        return (np.random.uniform(-width, width), np.random.uniform(-height, height))

    def distance(p1,p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    tree_locations = []

    while len(tree_locations) < nb_trees:
        newtree = gentree()
        if valid_check(newtree):
            for tree in tree_locations:
                if distance(newtree,tree) < radius: break
            else:
                tree_locations.append(newtree)

    return tree_locations
