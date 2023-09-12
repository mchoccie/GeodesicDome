Module geo_dome.tessellation
============================

Functions
---------

    
`add_middle_get_index(matrix: numpy.ndarray, new_vertices: numpy.ndarray, old_vertices: numpy.ndarray, v_index: numpy.int64, v1: numpy.int64, v2: numpy.int64) ‑> numpy.int64`
:   Matrix based method for inserting new vertices with the guarantee of uniqueness
    
    Args:
        matrix (np.ndarray): (Nvertices x Nvertices) matrix 
        new_vertices (np.ndarray): Array to store new vertices 
        old_vertices (np.ndarray): Array of old vertices
        v_index (np.int64): Current index of new vertices
        v1 (np.int64): Index of outer vertex
        v2 (np.int64): Index of other outer vertex
    
    Returns:
        Union[np.int64, np.int64]: index of inserted vertex, new v_index

    
`calc_dist(points) ‑> list`
:   Calculates the distance of each point in the Dome from the origin
    
    Args:
        points (list): list of points
    
    Returns:
        list: List of distances of each point in the Dome

    
`create_geodesic_dome(freq=0) ‑> numpy.ndarray`
:   Creates an geodesic dome of a given frequency
    
    Args:
        freq (int, optional): the frequency of the dome. Defaults to 0.
    
    Returns:
        Union[np.ndarray, np.ndarray]: the array of vertices, the array of
        triangles
    
        Vertices = [[x,y,z], ... , [x,y,z]]
        Triangles = [[v1, v2, v3], ...] where vx is the index of a vertex in the vertices array
    
        Adjacency list = [[v1, ..., v5, v6?], ...] where vx is the index of a vertex. v6 may not exist for some vertices

    
`find_adjacent_triangles(triangles: numpy.ndarray, vertices: numpy.ndarray) ‑> numpy.ndarray`
:   Finds adjacent triangles for a given point
    
    Args:
        triangles (np.ndarray): The list of triangles in the dome
        vertices (np.ndarray): The vertices (index) to find the adjacent triangles for
    
    Returns:
        np.ndarray: List of indices for triangles in the array

    
`get_middle_coords(v1: numpy.ndarray, v2: numpy.ndarray) ‑> numpy.ndarray`
:   Gets the midpoint between two coords
    
    Args:
        v1 (np.ndarray): coord 1
        v2 (np.ndarray): coord 2
    
    Returns:
        np.ndarray: the midpoint (not normalised)

    
`is_zero(coord, message='') ‑> None`
:   Debug method, checks if a coord is at origin
    
    Args:
        coord (np array): coordinate to check
        message (str, optional): descriptive message when triggered. Defaults to "".

    
`normalise_all(new_vertices: numpy.ndarray) ‑> None`
:   Normalises all the vertices in an array
    
    Args:
        new_vertices (np.ndarray): the array of vertices

    
`normalise_length(coords: numpy.ndarray) ‑> numpy.ndarray`
:   Normalises the distance from origin of a coord. Multiplies by the
    frequency the icosphere to avoid floating point precision errors
    
    Args:
        coords (np.ndarray): coordinate to normalise
    
    Returns:
        np.ndarray: normalised coordinate

    
`tessellate_geodesic_dome(vertices: numpy.ndarray, triangles: numpy.ndarray, target_triangles: numpy.ndarray = None) ‑> numpy.ndarray`
:   Tessellates the target triangles in the dome
    
    Args:
        vertices (np.ndarray): Array of vertices
        triangles (np.ndarray): Array of triangles
        target_triangles (np.ndarray): Target triangles to tessellate 
                                        (optional: tessellates all triangles if not given)
    
    Returns:
        Union[np.ndarray, np.ndarray, np.ndarray]: Array of new vertices,
        triangles, and new adjacency list