# import geo_dome.numba_geodesic_dome as gd
# import geo_dome.numba_geodesic_dome_private as priv
from geo_dome.tessellation import *
from geo_dome.neighbourhood_search import *

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


class TestNeighbourhoodSearch(unittest.TestCase):
    def test_adj_insert(self):
        vertices, triangles, adj_list = create_geodesic_dome.py_func(0)
        adj = np.full((len(vertices), 6), -1, dtype=np.int64)

        adj_insert(adj, triangles[0][0], triangles[0][1])
        adj_insert(adj, triangles[0][0], triangles[0][2])
        adj_insert(adj, triangles[0][1], triangles[0][0])
        adj_insert(adj, triangles[0][1], triangles[0][2])
        adj_insert(adj, triangles[0][2], triangles[0][0])
        adj_insert(adj, triangles[0][2], triangles[0][1])

        self.assertNotEqual(adj.shape[0], 0)

    def test_adj_insert2(self):
        vertices, triangles, adj_list = create_geodesic_dome.py_func(0)
        adj = np.full((len(vertices), 6), -1, dtype=np.int64)

        adj_insert(adj, triangles[0][0], triangles[0][1])
        adj_insert(adj, triangles[0][0], triangles[0][2])
        adj_insert(adj, triangles[0][1], triangles[0][0])
        adj_insert(adj, triangles[0][1], triangles[0][2])
        adj_insert(adj, triangles[0][2], triangles[0][0])
        adj_insert(adj, triangles[0][2], triangles[0][1])

        self.assertEqual(adj.shape[0], 12)

    def test_always_six_vertices(self):
        vertices, triangles, adj_list = create_geodesic_dome.py_func(0)
        adj = np.full((len(vertices), 6), -1, dtype=np.int64)

        adj_insert(adj, triangles[0][0], triangles[0][1])
        adj_insert(adj, triangles[0][0], triangles[0][2])
        adj_insert(adj, triangles[0][1], triangles[0][0])
        adj_insert(adj, triangles[0][1], triangles[0][2])
        adj_insert(adj, triangles[0][2], triangles[0][0])
        adj_insert(adj, triangles[0][2], triangles[0][1])

        self.assertEqual(adj.shape[1], 6)

    def test_last_always_negative_vertices(self):
        vertices, triangles, adj_list = create_geodesic_dome.py_func(0)
        adj = np.full((len(vertices), 6), -1, dtype=np.int64)

        adj_insert(adj, triangles[0][0], triangles[0][1])
        adj_insert(adj, triangles[0][0], triangles[0][2])
        adj_insert(adj, triangles[0][1], triangles[0][0])
        adj_insert(adj, triangles[0][1], triangles[0][2])
        adj_insert(adj, triangles[0][2], triangles[0][0])
        adj_insert(adj, triangles[0][2], triangles[0][1])

        self.assertEqual(adj[0][5], -1)

    def test_always_correct_neighbours(self):
        vertices, triangles, adj_list = create_geodesic_dome.py_func(0)
        adj = np.full((len(vertices), 6), -1, dtype=np.int64)

        adj_insert(adj, triangles[0][0], triangles[0][1])
        adj_insert(adj, triangles[0][0], triangles[0][2])
        adj_insert(adj, triangles[0][1], triangles[0][0])
        adj_insert(adj, triangles[0][1], triangles[0][2])
        adj_insert(adj, triangles[0][2], triangles[0][0])
        adj_insert(adj, triangles[0][2], triangles[0][1])

        self.assertEqual(adj[0][0], 11)
        self.assertEqual(adj[0][1], 5)
        self.assertEqual(adj[0][2], -1)
        self.assertEqual(adj[0][3], -1)

    def test_create_adj_lists_not_none(self):
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
        icosa_triangles = np.array(
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
        adj_list = create_adj_list.py_func(icosa_vertices, icosa_triangles)

        self.assertIsNotNone(adj_list)

    def test_create_correct_shape(self):
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
        icosa_triangles = np.array(
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
        adj_list = create_adj_list.py_func(icosa_vertices, icosa_triangles)

        self.assertEqual(adj_list.shape[0], 12)

    def test_create_correct_shape_0_vertices(self):
        g_ratio = (1 + math.sqrt(5)) / 2
        # creating initial icosahedron vertices
        icosa_vertices = np.array(
            [],
            dtype=np.float64,
        )
        # creating initial icosahedron edges
        icosa_triangles = np.array(
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
        adj_list = create_adj_list.py_func(icosa_vertices, icosa_triangles)

        self.assertEqual(adj_list.shape[0], 0)

    def test_correct_values(self):
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
        icosa_triangles = np.array(
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

        adj_list = create_adj_list.py_func(icosa_vertices, icosa_triangles)

        self.assertEqual(adj_list[3][2], 2)
        self.assertEqual(adj_list[5][3], 9)
        self.assertEqual(adj_list[8][4], 9)

    def test_always_six_for_inner_shape(self):
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
        icosa_triangles = np.array(
            [
                (0, 11, 5),
                (0, 5, 1),
            ],
            dtype=np.int64,
        )

        adj_list = create_adj_list.py_func(icosa_vertices, icosa_triangles)

        self.assertEqual(adj_list.shape[1], 12)

    def test_last_is_always_negative(self):
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
        icosa_triangles = np.array(
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

        adj_list = create_adj_list.py_func(icosa_vertices, icosa_triangles)
        for row in adj_list:
            self.assertEqual(-1, row[5])

    def test_neighbourhood_correctness_numba(self):
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

        icosa_triangles = np.array(
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

        adj_list = create_adj_list.py_func(icosa_vertices, icosa_triangles)
        neighbours = find_neighbours_vertex.py_func(icosa_vertices, adj_list, 0, 1)
        self.assertEqual(neighbours[0], 0)
        self.assertEqual(neighbours[1], 11)
        self.assertEqual(neighbours[2], 5)
        self.assertEqual(neighbours[3], 1)
        self.assertEqual(neighbours[4], 7)
        self.assertEqual(neighbours[5], 10)

    def test_neighbourhood_triangles(self):
        start_vertices = np.zeros(3, dtype=np.int64)
        vertices, triangles, adj = create_geodesic_dome.py_func()
        triangle = triangles[1]
        for i in range(len(triangle)):
            start_vertices[i] = triangle[i]

        neighbours = find_neighbours_triangle.py_func(
            vertices, adj, start_vertices, 1
        )

        self.assertEqual(neighbours[0], 0);
        self.assertEqual(neighbours[1], 5);
