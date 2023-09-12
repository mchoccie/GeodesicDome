Module geo_dome.neighbourhood_search
====================================

Functions
---------

    
`adj_insert(adj: numpy.ndarray, root: numpy.int64, neighbour: numpy.int64) ‑> None`
:   Function to insert a point into adjacency list of root vertex
    
    Args:
        adj (np.ndarray): array of arrays, representing adjacency list
        root (np.int64): index of root vertex
        neighbour (np.int64): index of neighbour vertex to add

    
`create_adj_list(vertices: numpy.ndarray, triangles: numpy.ndarray) ‑> numpy.ndarray`
:   Function to create adjacency list representation of vertices
    
    Args:
        vertices (np.ndarray): numpy array of vertices
        triangles (np.ndarray): numpy array of vertices
    
    Returns:
        np.ndarray: array of arrays representing adjacency list

    
`find_neighbours_triangle(vertices: numpy.ndarray, adj_list: numpy.ndarray, start_vertices: numpy.ndarray, depth=1) ‑> numpy.ndarray`
:   Function to find nearest neighbours to a specific point, up to a
    specified depth
    
    Args:
        vertices (np.ndarray): numpy array of vertices in the Dome
        adj (np.ndarray): adjacency list of the vertices
        index (np.int64): index of the root triangle
        depth (np.int64, optional): search depth. Defaults to 1.
    
    Returns:
        np.ndarray: Array of neighbours found, may include -1 representing empty entries

    
`find_neighbours_vertex(vertices: numpy.ndarray, adj_list: numpy.ndarray, index: numpy.int64, depth=1) ‑> numpy.ndarray`
:   Function to find nearest neighbours to a specific point, up to a
    specified depth
    
    Args:
        vertices (np.ndarray): numpy array of vertices in the Dome
        adj (np.ndarray): adjacency list of the vertices
        index (np.int64): index of the root vertex
        depth (np.int64, optional): search depth. Defaults to 1.
    
    Returns:
        np.ndarray: Array of neighbours found, may include -1 representing empty entries