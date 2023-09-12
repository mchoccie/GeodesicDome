Module geo_dome.geodesic_dome
=============================

Classes
-------

`GeodesicDome(freq=0)`
:   Class wrapper to create and interact with a Geodesic Dome
    
    Creates a given geodesic dome with a given frequency.
    
    Args:
        freq (int, optional): The frequency of the geodesic dome. Defaults to 0.

    ### Methods

    `custom_partial_tessellate_triangle(self, target_triangles: numpy.ndarray) ‑> None`
    :   Selectively tessellate certain triangles in the Geodesic Dome
        
        Args:
            target_triangles (np.ndarray): indices of triangles to tessellate
        
        Raises:
            ValueError: Target triangles not provided

    `custom_partial_tessellate_vertex(self, neighbours=array([], dtype=int64)) ‑> None`
    :   Tessellates all adjacent triangles to a given set of vertices. If not vertices are given,
        it will attempt to tessellate from the most recent neighbourhood search results
        
        Args:
            neighours (np.ndarray, optional): The set of vertices to tessellate
            around (provided as indices). Defaults to an empty array.

    `find_neighbours_triangle(self, index: numpy.int64, depth=0) ‑> numpy.ndarray`
    :   Finds the neighbours of a given triangle's vertices on the geodesic
        dome to a certain depth (defaults to 0 if not provided)
        
        Args:
            index (np.int64): index of root triangle
            depth (int, optional): tessellation depth. Defaults to 0.
        
        Raises:
            ValueError: Raised when depth is negative
            ValueError: Raised when triangle is out of bounds
        
        Returns:
            np.ndarray: An array containing the indices of all the triangle's neighbouring vertices

    `find_neighbours_vertex(self, index: numpy.int64, depth=0) ‑> numpy.ndarray`
    :   Finds the neighbours of a given vertex on the geodesic dome to a certain depth (defaults to 0 if not provided)
        
        Args:
            index (np.int64): The index of the vertex to search from
            depth (int, optional): The depth of neighbours to return. Defaults to 1.
        
        Raises:
            ValueError: Raised when depth is negative
            ValueError: Raised when vertex is out of bounds
        
        Returns:
            np.ndarray: An array containing the indices of all the vertex's
            neighbours

    `get_triangles(self) ‑> numpy.ndarray`
    :   Getter function for triangles
        
        Returns:
            np.ndarray: the triangles of the geodesic dome

    `get_vertices(self) ‑> numpy.ndarray`
    :   Getter function for vertices
        
        Returns:
            np.ndarray: the vertices of the geodesic dome

    `partial_tessellate_triangle(self, index: numpy.int64, depth=0) ‑> None`
    :   Main entrypoint to tessellate dome based on selected triangle
        
        Args:
            index (np.int64): index of root triangle
            depth (int, optional): tessellation depth. Defaults to 0.
        
        Raises:
            ValueError: Raised when depth is negative
            ValueError: Raised when triangle is out of bounds

    `partial_tessellate_vertex(self, index: numpy.int64, depth=0) ‑> None`
    :   Main entrypoint to tessellate dome based on selected vertex
        
        Args:
            index (np.int64): index of root vertex
            depth (int, optional): tessellation depth. Defaults to 0.
        
        Raises:
            ValueError: Raised when depth is negative
            ValueError: Raised when vertex is out of bounds

    `tessellate(self, freq=1) ‑> None`
    :   Tessellates the geodesic dome a given number of times (tessellates once if no arguments provided)
        
        Args:
            freq (int, optional): The number of times to tessellate. Defaults to 1.