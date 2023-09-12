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


# class TestTessellation(unittest.TestCase):
#     def test_tesselation(self):
#         g_ratio = (1 + math.sqrt(5)) / 2
#         # creating initial icosahedron vertices
#         icosa_vertices = np.array(
#             [
#                 (-1, g_ratio, 0),
#                 (1, g_ratio, 0),
#                 (-1, -(g_ratio), 0),
#                 (1, -(g_ratio), 0),
#                 (0, -1, g_ratio),
#                 (0, 1, g_ratio),
#                 (0, -1, -(g_ratio)),
#                 (0, 1, -(g_ratio)),
#                 (g_ratio, 0, -1),
#                 (g_ratio, 0, 1),
#                 (-(g_ratio), 0, -1),
#                 (-(g_ratio), 0, 1),
#             ],
#             dtype=np.float64,
#         )
#         # creating initial icosahedron edges
#         icosa_triangles = np.array(
#             [
#                 (0, 11, 5),
#                 (0, 5, 1),
#                 (0, 1, 7),
#                 (0, 7, 10),
#                 (0, 10, 11),
#                 (1, 5, 9),
#                 (5, 11, 4),
#                 (11, 10, 2),
#                 (10, 7, 6),
#                 (7, 1, 8),
#                 (3, 9, 4),
#                 (3, 4, 2),
#                 (3, 2, 6),
#                 (3, 6, 8),
#                 (3, 8, 9),
#                 (4, 9, 5),
#                 (2, 4, 11),
#                 (6, 2, 10),
#                 (8, 6, 7),
#                 (9, 8, 1),
#             ],
#             dtype=np.int64,
#         )

#         # Array for normalised vertices
#         icosa_vertices_normalised = np.zeros((len(icosa_vertices), 3), dtype=np.float64)

#         # Normalise all icosahedron vertices
#         for i in range(len(icosa_vertices)):
#             icosa_vertices_normalised[i] = normalise_length(icosa_vertices[i])

#         new_vertices, icosa_triangles = tessellate_geodesic_dome.py_func(
#             icosa_vertices_normalised, icosa_triangles, math.pow(4, 1)
#         )

#         #This will change just want to see coverage reports for now
#         self.assertEqual(1, 1)
