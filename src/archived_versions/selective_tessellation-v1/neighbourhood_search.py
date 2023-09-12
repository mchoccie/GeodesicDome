import numpy as np
from numba import njit


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
