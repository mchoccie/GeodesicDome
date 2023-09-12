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


class SelectiveTessellationTest(unittest.TestCase):

    def test_tessellate_one_icosahedron(self):      
        self.neighbours = np.zeros(0, dtype=np.int64)
        vertices, triangles, adj = create_geodesic_dome.py_func()

        target = np.ndarray(1, dtype=np.int64)
        target[0] = 0

        vertices, triangles, adj = tessellate_geodesic_dome.py_func(
            vertices, triangles, target)

        self.assertEqual(len(vertices), 15)
        self.assertEqual(len(triangles), 23)
        return

    def test_find_adjacent_triangle(self):
        vertices, triangles, adj = create_geodesic_dome()

        target = np.ndarray(1, dtype=np.int64)

        for i in range(12):
            target[0] = i
            found = find_adjacent_triangles.py_func(triangles, target)
            self.assertEqual(len(found), 5)
        return

    def test_find_adjacent_triangle_two_targets_no_overlapping_triangles(self):
        vertices, triangles, adj = create_geodesic_dome()

        target = np.ndarray(2, dtype=np.int64)

        
        target[0] = 3
        target[1] = 11
     
        found = find_adjacent_triangles.py_func(triangles, target)
        self.assertEqual(len(found), 10)
        return

    def test_find_adjacent_triangle_two_targets_overlapping_triangles(self):
        vertices, triangles, adj = create_geodesic_dome()

        target = np.ndarray(2, dtype=np.int64)

        
        target[0] = 3
        target[1] = 4
     
        found = find_adjacent_triangles.py_func(triangles, target)
        self.assertEqual(len(found), 8)
        return

    def test_find_adjacent_triangle_three_targets_overlapping_triangles(self):
        vertices, triangles, adj = create_geodesic_dome()

        target = np.ndarray(3, dtype=np.int64)

        
        target[0] = 3
        target[1] = 4
        target[2] = 6
     
        found = find_adjacent_triangles.py_func(triangles, target)
        self.assertEqual(len(found), 11)
        return