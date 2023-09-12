import math
import numpy as np
import numba
from numba import njit
from numba.typed import Dict
from typing import Union


SCALE = 1


@njit
def is_zero(coord, message="") -> None:
    """Debug method, checks if a coord is at origin

    Args:
        coord (np array): coordinate to check
        message (str, optional): descriptive message when triggered. Defaults to "".
    """
    if (coord == np.array([0, 0, 0])).all():
        print("Something is zero that shouldn't be: " + message)


@njit
def normalise_length(coords: np.ndarray) -> np.ndarray:
    """Normalises the distance from origin of a coord. Multiplies by the
    frequency the icosphere to avoid floating point precision errors

    Args:
        coords (np.ndarray): coordinate to normalise

    Returns:
        np.ndarray: normalised coordinate
    """
    length = math.sqrt(
        math.pow(coords[0], 2) + math.pow(coords[1], 2) +
        math.pow(coords[2], 2)
    )

    is_zero(coords, "normalise")

    return np.array(
        [
            (coords[0] / length) * SCALE,
            (coords[1] / length) * SCALE,
            (coords[2] / length) * SCALE,
        ]
    )


@njit
def get_middle_coords(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Gets the midpoint between two coords

    Args:
        v1 (np.ndarray): coord 1
        v2 (np.ndarray): coord 2

    Returns:
        np.ndarray: the midpoint (not normalised)
    """
    ret = np.array(
        [(v2[0] + v1[0]) / 2, (v2[1] + v1[1]) / 2, (v2[2] + v1[2]) / 2],
        dtype=np.float64,
    )
    return ret


@njit
def add_middle_to_vertices(mid: np.ndarray, vertices: np.ndarray, v_index: Dict) -> int:
    """Adds a given midpoint to a list of new vertices

    Args:
        mid (np.ndarray): the midpoint to add
        vertices (np.ndarray): an array of new vertices, to be concatenated with
        existing vertices
        v_index (Dict): dictionary containing the indexes of existing midpoints,
        to prevent duplicates

    Returns:
        int: the index of the midpoint that was added to vertices
    """
    # Creating (hopefully) unique key for each coordinate
    mid_sum = mid[0] * 3 + mid[1] * 2 + mid[2]
    # Adding key to dictionary of coords with index as value
    if mid_sum not in v_index:
        v_index[mid_sum] = len(v_index)
    index = v_index[mid_sum]
    # Add new midpoint to new vertices array
    if (vertices[index] == np.array([0, 0, 0])).all():
        vertices[index] = mid

    return index


@njit
def normalise_all(new_vertices: np.ndarray) -> None:
    """Normalises all the vertices in an array

    Args:
        new_vertices (np.ndarray): the array of vertices
    """
    for i in range(len(new_vertices)):
        new_vertices[i] = normalise_length(new_vertices[i])


@njit
def adj_insert(adj: np.ndarray, root: np.int64, neighbour: np.int64) -> None:
    """Function to insert a point into adjacency list of root vertex

    Args:
        adj (np.ndarray): array of arrays, representing adjacency list
        root (np.int64): index of root vertex
        neighbour (np.int64): index of neighbour vertex to add
    """
    root_list = adj[root]
    for i in range(6):
        if root_list[i] == neighbour:
            break
        if root_list[i] == -1:
            root_list[i] = neighbour
            break


@njit
def create_adj_list(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Function to create adjacency list representation of vertices

    Args:
        vertices (np.ndarray): numpy array of vertices
        triangles (np.ndarray): numpy array of vertices

    Returns:
        np.ndarray: array of arrays representing adjacency list
    """
    adj = np.full((len(vertices), 6), -1, dtype=np.int64)

    for t in triangles:
        adj_insert(adj, t[0], t[1])
        adj_insert(adj, t[0], t[2])
        adj_insert(adj, t[1], t[0])
        adj_insert(adj, t[1], t[2])
        adj_insert(adj, t[2], t[0])
        adj_insert(adj, t[2], t[1])

    return adj


@njit
def generate_new_points(
    vertices: np.ndarray, triangles: np.ndarray, t: int
) -> Union[np.ndarray, np.ndarray]:
    """Tesselates the entire icosphere once. Returns an array containing the
    new vertices, to be concatenated with existing vertices, and a set of new
    triangles, to replace the old triangles

    Args:
        vertices (np.ndarray): the current vertices in the icosphere
        triangles (np.ndarray): the current triangles in the icosphere
        t (int): triangulation factor, used to calculate sizes of new vertices and triangles arrays

    Returns:
        np.ndarray: array of new vertices
        np.ndarray: array of new triangles
    """
    # create new array for new triangles
    new_triangles = np.zeros((len(triangles) * 4, 3), dtype=numba.int64)
    n_old_vertices = 10 * int(t / 4) + 2
    n_new_vertices = 10 * int(t - t / 4)
    # create new array for new vertices
    new_vertices = np.zeros((n_new_vertices, 3), dtype=numba.float64)

    i = 0
    v_index = Dict.empty(
        key_type=np.float64,
        value_type=np.int64,
    )
    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        is_zero(v0, "vertex")
        is_zero(v1, "vertex")
        is_zero(v2, "vertex")

        # Get midpoints for each edge of the triangle
        mid01 = get_middle_coords(v0, v1)
        mid12 = get_middle_coords(v1, v2)
        mid02 = get_middle_coords(v2, v0)

        # Get indexes of the new midpoints with respect to current vertices
        index01 = add_middle_to_vertices(
            mid01, new_vertices, v_index) + n_old_vertices
        index12 = add_middle_to_vertices(
            mid12, new_vertices, v_index) + n_old_vertices
        index02 = add_middle_to_vertices(
            mid02, new_vertices, v_index) + n_old_vertices

        # Create new triangles
        new_triangles[i] = [tri[0], index01, index02]
        new_triangles[i + 1] = [tri[1], index12, index01]
        new_triangles[i + 2] = [tri[2], index02, index12]
        new_triangles[i + 3] = [index01, index12, index02]

        i += 4
    return new_vertices, new_triangles


@njit
def calc_dist(points) -> list:
    """Calculates the distance of each point in the Dome from the origin

    Args:
        points (list): list of points

    Returns:
        list: List of distances of each point in the Dome
    """

    distances = []
    for p in points:
        dist = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        distances.append(round(dist, 2))

    return distances
