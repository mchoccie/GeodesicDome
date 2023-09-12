import math
import numpy as np
from numba import njit
from numba.typed import Dict
from typing import Union

# try:
#     from geo_dome.neighbourhood_search import *
# except:
# from neighbourhood_search import *

from neighbourhood_search import *

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
def create_geodesic_dome(freq=0) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """Creates an geodesic dome of a given frequency

    Args:
        freq (int, optional): the frequency of the dome. Defaults to 0.

    Returns:
        Union[np.ndarray, np.ndarray]: the array of vertices, the array of
        triangles

        Vertices = [[x,y,z], ... , [x,y,z]]
        Triangles = [[v1, v2, v3], ...] where vx is the index of a vertex in the vertices array

        Adjacency list = [[v1, ..., v5, v6?], ...] where vx is the index of a vertex. v6 may not exist for some vertices
    """
    # Set normalised scaling
    if freq != 0:
        SCALE = freq

    g_ratio = (1 + math.sqrt(5)) / 2

    # creating initial icosahedron vertices
    icosa_vertices = np.array(
        [
            (-1, g_ratio, 0),
            (1, g_ratio, 0),
            (-1, -(g_ratio), 0),
            (1, -(g_ratio), 0),
            (0, -1, g_ratio),
            (0, 1, g_ratio),
            (0, -1, -(g_ratio)),
            (0, 1, -(g_ratio)),
            (g_ratio, 0, -1),
            (g_ratio, 0, 1),
            (-(g_ratio), 0, -1),
            (-(g_ratio), 0, 1),
        ],
        dtype=np.float64,
    )
    # creating initial icosahedron edges
    triangles = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )

    # Array for normalised vertices
    vertices = np.zeros((len(icosa_vertices), 3), dtype=np.float64)

    # Normalise all icosahedron vertices
    for i in range(len(icosa_vertices)):
        vertices[i] = normalise_length(icosa_vertices[i])
    # Tessellate icosahedron
    adj_list = create_adj_list(vertices, triangles)
    for i in range(freq):
        vertices, triangles, adj_list = tessellate_geodesic_dome(
            vertices, triangles)
    return vertices, triangles, adj_list


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


@njit
def tessellate_geodesic_dome(
    vertices, triangles
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """Tesselates the entire icosphere once. Returns an array containing the
    new vertices, to be concatenated with existing vertices, and a set of new
    triangles, to replace the old triangles

    Args:
        vertices (np.ndarray): the current vertices in the icosphere
        triangles (np.ndarray): the current triangles in the icosphere

    Returns:
        np.ndarray: array of new vertices
        np.ndarray: array of new triangles
        np.ndarray: array of arrays representing adjacency list
    """
    t = ((len(vertices) - 2) / 10) * 4

    # create new array for new triangles
    new_triangles = np.zeros((len(triangles) * 4, 3), dtype=np.int64)
    n_old_vertices = len(vertices)
    n_new_vertices = 10 * int(t - t / 4)
    # create new array for new vertices
    new_vertices = np.zeros((n_new_vertices, 3), dtype=np.float64)

    i = 0
    v_index = Dict.empty(
        key_type=np.float64,
        value_type=np.int64,
    )

    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

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
    normalise_all(new_vertices)
    # Keep track of all previous vertices
    old_vertices = vertices
    # Create array to concatenate old vertices with new midpoints
    vertices = np.zeros(
        (len(old_vertices) + len(new_vertices), 3), dtype=np.float64)

    i = 0
    # Add old vertices
    for v in old_vertices:
        vertices[i] = v
        i += 1
    # Add new midpoints
    for v in new_vertices:
        vertices[i] = v
        i += 1

    new_adj_list = create_adj_list(vertices, new_triangles)
    return vertices, new_triangles, new_adj_list
