import numpy as np
from numba import njit
import numba
from numba.typed import Dict
from typing import Union

# try:
#     from geo_dome.tessellation import *
#     from geo_dome.neighbourhood_search import *
# except:
#     from tessellation import *
#     from neighbourhood_search import *
from tessellation import *
from neighbourhood_search import *


@njit
def find_adjacent_triangles(triangles: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Finds adjacent triangles for a given point

    Args:
        triangles (np.ndarray): The list of triangles in the dome
        vertices (np.ndarray): The vertices (index) to find the adjacent triangles for

    Returns:
        np.ndarray: List of indices for triangles in the array
    """
    found_triangles = np.zeros((len(triangles)), dtype=np.int64)
    found_index = 0

    for i in range(len(triangles)):
        t = triangles[i]
        if t[0] in vertices or t[1] in vertices or t[2] in vertices:
            found_triangles[found_index] = i
            found_index += 1

    found_triangles = np.resize(found_triangles, found_index)

    return found_triangles


@njit
def check_edge(
    v1: int, v2: int, edge_matrix: np.ndarray, edge_count: int
) -> Union[np.ndarray, int]:
    """Checks if current edge has been previously visited

    Args:
        v1 (int): First vertex
        v2 (int): Second vertex
        edge_matrix (np.ndarray): 2D Matrix of all visited edges in dome
        edge_count (int): Number of edges found

    Returns:
        Union[np.ndarray, int]: Updated edge matrix, updated edge count
    """

    if edge_matrix[v1][v2] == -1:
        edge_count += 1
        edge_matrix[v1][v2] = 1
        edge_matrix[v2][v1] = 1

    return edge_matrix, edge_count


@njit
def calculate_edges(
    vertices: np.ndarray, triangles: np.ndarray, target_triangles: np.ndarray
) -> int:
    """Finds all edges connected to the target triangles

    Args:
        vertices (np.ndarray): Vertex array
        triangles (np.ndarray): Triangles array
        target_triangles (np.ndarray): Array of triangles the find connected
        edges to

    Returns:
        int: Number of edges found
    """
    edge_matrix = np.full((len(vertices), len(vertices)), -1, dtype=np.int64)
    edge_count = 0

    for i in target_triangles:
        t = triangles[i]
        edge_matrix, edge_count = check_edge(
            t[0], t[1], edge_matrix, edge_count)
        edge_matrix, edge_count = check_edge(
            t[1], t[2], edge_matrix, edge_count)
        edge_matrix, edge_count = check_edge(
            t[0], t[2], edge_matrix, edge_count)

    # np.resize(edges, (edge_count, 2))

    return edge_count


@njit
def tessellate_full_through_selective(
    vertices: np.ndarray, triangles: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """Tessellates the entire geodesic dome

    Args:
        vertices (np.ndarray): Array of vertices
        triangles (np.ndarray): Array of triangles

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Array of new vertices,
        triangles, and new adjacency list
    """
    target = np.zeros((len(triangles)), dtype=np.int64)

    for i in range(len(triangles)):
        target[i] = i

    return selective_tessellation(vertices, triangles, target)


@njit
def add_middle_get_index(matrix: np.ndarray, new_vertices: np.ndarray,
                         old_vertices: np.ndarray, v_index: np.int64, v1: np.int64, v2: np.int64) -> Union[np.int64, np.int64]:
    """Matrix based method for inserting new vertices with the guarantee of uniqueness

    Args:
        matrix (np.ndarray): (Nvertices x Nvertices) matrix 
        new_vertices (np.ndarray): Array to store new vertices 
        old_vertices (np.ndarray): Array of old vertices
        v_index (np.int64): Current index of new vertices
        v1 (np.int64): Index of outer vertex
        v2 (np.int64): Index of other outer vertex

    Returns:
        Union[np.int64, np.int64]: index of inserted vertex, new v_index
    """
    if (matrix[v1][v2] == -1):
        matrix[v1][v2] = v_index
        matrix[v2][v1] = v_index
        new_vertices[v_index] = get_middle_coords(
            old_vertices[v1], old_vertices[v2])
        v_index += 1

    return matrix[v1][v2], v_index


@njit
def selective_tessellation(
    vertices: np.ndarray, triangles: np.ndarray, target_triangles: np.ndarray
) -> Union[np.ndarray, np.ndarray]:
    """Tessellates the target triangles in the dome

    Args:
        vertices (np.ndarray): Array of vertices
        triangles (np.ndarray): Array of triangles
        target_triangles (np.ndarray): Target triangles to tessellate

    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Array of new vertices,
        triangles, and new adjacency list
    """
    # First calculate the number of edges covered by the current triangles

    edge_count = calculate_edges(vertices, triangles, target_triangles)

    # create new array for new triangles
    n_new_triangles = len(triangles) + 3 * len(target_triangles)
    new_triangles = np.zeros((n_new_triangles, 3), dtype=np.int64)
    n_old_vertices = len(vertices)
    n_new_vertices = edge_count
    # create new array for new vertices
    new_vertices = np.zeros((n_new_vertices, 3), dtype=np.float64)
    vertex_matrix = np.full((len(vertices), len(vertices)), -1, dtype=np.int64)
    v_index = 0

    i = 0

    hit_triangles = np.zeros((len(triangles)), dtype=np.int8)

    for t in target_triangles:
        tri = triangles[t]
        hit_triangles[t] = 1

        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        # Get midpoints for each edge of the triangle
        index01, v_index = add_middle_get_index(
            vertex_matrix, new_vertices, vertices, v_index, tri[0], tri[1])
        index12, v_index = add_middle_get_index(
            vertex_matrix, new_vertices, vertices, v_index, tri[1], tri[2])
        index02, v_index = add_middle_get_index(
            vertex_matrix, new_vertices, vertices,  v_index, tri[0], tri[2])

        index01 += n_old_vertices
        index12 += n_old_vertices
        index02 += n_old_vertices

        # Create new triangles
        new_triangles[i] = [tri[0], index01, index02]
        new_triangles[i + 1] = [tri[1], index12, index01]
        new_triangles[i + 2] = [tri[2], index02, index12]
        new_triangles[i + 3] = [index01, index12, index02]

        i += 4

    # Add all the untargeted triangles
    for j in range(len(triangles)):
        if hit_triangles[j] == 0:
            t = triangles[j]
            new_triangles[i] = t
            i += 1

    # print(new_vertices)

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
