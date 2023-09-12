import math

# http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
# Basic icosahedron construction code credited to Andreas Kahler (20th June,
# 2009)


def get_middle_coords(v1, v2):
    """Gets the middle of two coords

    Args:
        v1 (tuple): The first coord
        v2 (tuple): The second coord

    Returns:
        tuple: middle coord
    """
    ret = ((v2[0] + v1[0]) / 2, (v2[1] + v1[1]) / 2, (v2[2] + v1[2]) / 2)
    return ret


def add_middle_to_vertices(mid, vertices, v_index):
    """Adds a coord to vertices and returns its index
    (only really used in the tessellation alg)

    Args:
        mid (tuple): new coord to add
        vertices (list): list of vertices
        v_index (dict): dictionary mapping coords to their index in vertices

    Returns:
        int: index of added coord (or existing dup coord)
    """
    if mid not in v_index:
        vertices.append(normalise_length(mid))
        v_index[mid] = len(vertices) - 1

    return v_index[mid]


def tessellate_and_normalise(vertices, triangles):
    """Tesselates the entire icosphere

    Args:
        vertices (list): list of vertices
        triangles (set): set of triangles

    Returns:
        list, set: new vertices and new triangles
    """
    # Dictionary mapping coordinates to their index (key: tup, val: int)
    v_index = {}
    new_triangles = set()
    for tri in triangles:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]

        mid01 = get_middle_coords(v0, v1)
        mid12 = get_middle_coords(v1, v2)
        mid20 = get_middle_coords(v2, v0)

        index01 = add_middle_to_vertices(mid01, vertices, v_index)
        index12 = add_middle_to_vertices(mid12, vertices, v_index)
        index20 = add_middle_to_vertices(mid20, vertices, v_index)

        new_triangles.add((tri[0], index01, index20))
        new_triangles.add((tri[1], index12, index01))
        new_triangles.add((tri[2], index20, index12))
        new_triangles.add((index01, index12, index20))

    triangles = new_triangles

    return vertices, triangles


def normalise_length(coords):
    """Normalises the distance from origin of a coord

    Args:
        coords (tuple): coord

    Returns:
        tuple: normalised coord
    """
    length = math.sqrt(
        math.pow(coords[0], 2) + math.pow(coords[1], 2) + math.pow(coords[2], 2)
    )
    return (
        coords[0] / length,
        coords[1] / length,
        coords[2] / length,
    )


def create_adjacency_list_from_triangles(vertices, triangles):
    """Creates adjacency list for a given icosphere
    Adj list is separate list, with each index corresponding to the vertex
    with same index in vertices.

    Args:
        vertices (list): list of vertices
        triangles (set): set of triangles

    Returns:
        list(set): adjacency list of neighbour vertices for each vertex
    """
    adj = [set() for _ in range(len(vertices))]
    for t in triangles:
        adj[t[0]].add(t[1])
        adj[t[0]].add(t[2])
        adj[t[1]].add(t[0])
        adj[t[1]].add(t[2])
        adj[t[2]].add(t[0])
        adj[t[2]].add(t[1])

    return adj


def create_icosphere(freq=0):
    """Generates an icosphere of a given tessellation freq

    Args:
        freq (int, optional): the frequency of tessellation. Defaults to 0.

    Returns:
        list, set, list: vertices list, triangles set,  adj list
    """
    g_ratio = (1 + math.sqrt(5)) / 2

    icosa_vertices = [
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
    ]

    icosa_triangles = {
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
    }

    icosa_vertices_normalised = []
    for vertex in icosa_vertices:
        icosa_vertices_normalised.append(normalise_length(vertex))

    for i in range(freq):
        icosa_vertices_normalised, icosa_triangles = tessellate_and_normalise(
            icosa_vertices_normalised, icosa_triangles
        )

    icosa_adj = create_adjacency_list_from_triangles(
        icosa_vertices_normalised, icosa_triangles
    )

    return icosa_vertices_normalised, icosa_triangles, icosa_adj


def create_js_json(vertices: list, triangles: set) -> None:
    """Generates a json file for 3js visualisation from an icosphere

    Args:
        vertices (list): the vertices list
        triangles (set): the triangles set
    """
    f = open("3js/icojson.js", "w")
    f.write('export default {\n  "vertices": [\n')
    for vertex in vertices:
        for coord in vertex:
            f.write(f"{coord},")
    f.write('], "indices": [\n')
    for triangle in triangles:
        for coord in triangle:
            f.write(f"{coord},")
    f.write('], "radius": 1,')
    f.write('"details": 0}')


def main():
    vertices, triangles, edges, adj = create_icosphere(10)
    create_js_json(vertices, triangles)


if __name__ == "__main__":
    main()
