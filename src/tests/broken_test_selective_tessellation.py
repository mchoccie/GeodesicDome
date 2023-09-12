# import geo_dome.numba_geodesic_dome as gd
# import geo_dome.numba_geodesic_dome_private as priv
from geo_dome.tessellation import *
from geo_dome.neighbourhood_search import *
from geo_dome.selective_tessellation import *

import numpy as np
import io
import numba
import math
from numba.typed import Dict

import unittest.mock as mock
try:
    import unittest2 as unittest
except ImportError:
    import unittest


class SelectiveTessellationTest(unittest.TestCase):

    def test_tessellate_one_icosahedron(self):
        vertices, triangles, adj = create_geodesic_dome()

        target = np.ndarray(1, dtype=np.int64)
        target[0] = 0

        vertices, triangles, adj = selective_tessellation.py_func(
            vertices, triangles, target)

        self.assertEqual(len(vertices), 15)
        self.assertEqual(len(triangles), 23)
        return

    def test_tessellate_full_icosahedron(self):
        vertices, triangles, adj = create_geodesic_dome()

        # vertices, triangles, adj = tessellate_full_through_selective.py_func(
        #     vertices, triangles)
        vertices, triangles, adj = selective_tessellation.py_func(
            vertices, triangles)

        self.assertEqual(len(vertices), 42)
        self.assertEqual(len(triangles), 80)
        return

    def test_find_adjacent_triangle(self):
        vertices, triangles, adj = create_geodesic_dome()

        target = np.ndarray(1, dtype=np.int64)

        for i in range(12):
            target[0] = i
            found = find_adjacent_triangles.py_func(triangles, target)
            self.assertEqual(len(found), 5)
        return

    # def test_check_edge(self):
    #     vertices, triangles, adj = create_geodesic_dome()

    #     edge_matrix = np.full(
    #         (len(vertices), len(vertices)), -1, dtype=np.int64)
    #     edge_count = 0

    #     edge_matrix, edge_count = check_edge.py_func(
    #         0, 1, edge_matrix, edge_count)

    #     self.assertEqual(edge_count, 1)

    #     edge_matrix, edge_count = check_edge.py_func(
    #         0, 1, edge_matrix, edge_count)
    #     edge_matrix, edge_count = check_edge.py_func(
    #         1, 0, edge_matrix, edge_count)

    #     self.assertEqual(edge_count, 1)
    #     return

    # def test_calculate_edges(self):
    #     vertices, triangles, adj = create_geodesic_dome()

    #     target = np.ndarray(2, dtype=np.int64)
    #     target[0] = 0
    #     target[1] = 7

    #     edge_count = calculate_edges.py_func(vertices, triangles, target)

    #     self.assertEqual(edge_count, 6)
