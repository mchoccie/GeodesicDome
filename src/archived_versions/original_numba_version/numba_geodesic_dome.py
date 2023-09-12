import math
import numpy as np
from numba import njit
from numba.typed import Dict
from typing import Union

"""Import statement for library uploading"""
try:
    from geo_dome.unused import numba_geodesic_dome_private as priv
except:
    import numba_geodesic_dome_private as priv


@njit
def find_neighbours(
    vertices: np.ndarray, adj_list: np.ndarray, index: np.int64, depth=1
) -> np.ndarray:
    """Function to find nearest neighbours to a specific point, up to a
    specified depth

    Args:
        vertices (np.ndarray): numpy array of vertices in the Dome
        adj (np.ndarray): adjacency list of the vertices
        index (np.int64): index of the root vertex
        depth (np.int64, optional): search depth. Defaults to 1.

    Returns:
        np.ndarray: Array of neighbours found, may include -1 representing empty entries
    """
    size = 0
    for i in range(depth):
        size += (i + 1) * 6

    if size > len(vertices) - 1:
        size = len(vertices) - 1

    curr_depth = 1

    neighbours = np.full(size, -1, dtype=np.int64)
    num_neighbours = 0
    queue = np.full(1, index, dtype=np.int64)
    visited = np.full(len(vertices), False, dtype=np.bool_)

    q_end = 1

    while curr_depth <= depth:
        temp = np.full(curr_depth * 6, -1, dtype=np.int64)
        temp_ptr = 0
        q_front = 0

        while q_front < q_end:
            v_index = queue[q_front]
            for neighbour in adj_list[v_index]:
                if neighbour != -1 and visited[neighbour] == False:
                    neighbours[num_neighbours] = neighbour
                    temp[temp_ptr] = neighbour

                    num_neighbours += 1
                    temp_ptr += 1
                    visited[neighbour] = True
            visited[v_index] = True
            q_front += 1

        queue = temp
        q_end = temp_ptr
        curr_depth += 1
        if temp_ptr == 0:
            break

    return neighbours


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
        mid01 = priv.get_middle_coords(v0, v1)
        mid12 = priv.get_middle_coords(v1, v2)
        mid02 = priv.get_middle_coords(v2, v0)

        # Get indexes of the new midpoints with respect to current vertices
        index01 = (
            priv.add_middle_to_vertices(mid01, new_vertices, v_index) + n_old_vertices
        )
        index12 = (
            priv.add_middle_to_vertices(mid12, new_vertices, v_index) + n_old_vertices
        )
        index02 = (
            priv.add_middle_to_vertices(mid02, new_vertices, v_index) + n_old_vertices
        )

        # Create new triangles
        new_triangles[i] = [tri[0], index01, index02]
        new_triangles[i + 1] = [tri[1], index12, index01]
        new_triangles[i + 2] = [tri[2], index02, index12]
        new_triangles[i + 3] = [index01, index12, index02]

        i += 4
    priv.normalise_all(new_vertices)
    # Keep track of all previous vertices
    old_vertices = vertices
    # Create array to concatenate old vertices with new midpoints
    vertices = np.zeros((len(old_vertices) + len(new_vertices), 3), dtype=np.float64)

    i = 0
    # Add old vertices
    for v in old_vertices:
        vertices[i] = v
        i += 1
    # Add new midpoints
    for v in new_vertices:
        vertices[i] = v
        i += 1

    new_adj_list = priv.create_adj_list(vertices, new_triangles)
    return vertices, new_triangles, new_adj_list


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
        vertices[i] = priv.normalise_length(icosa_vertices[i])
    # Tessellate icosahedron
    adj_list = priv.create_adj_list(vertices, triangles)
    for i in range(freq):
        vertices, triangles, adj_list = tessellate_geodesic_dome(vertices, triangles)
    return vertices, triangles, adj_list
