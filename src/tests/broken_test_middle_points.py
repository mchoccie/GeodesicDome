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


class AddMiddlePointsTest(unittest.TestCase):
    def test_add_first_point(self):
        dictionary = Dict.empty(
            key_type=numba.float64,
            value_type=numba.int64,
        )
        first_point = np.array([1.2, 1.4, 1.6], dtype=np.float64)
        key = first_point[0] * 3 + first_point[1] * 2 + first_point[2]
        index = add_middle_get_index.py_func(
            first_point,
            np.zeros((2, 3), dtype=np.float64),
            dictionary,
        )
        self.assertEqual(index, 0)
        self.assertEqual(dictionary[key], 0)

    def test_add_another_point(self):
        dictionary = Dict.empty(
            key_type=numba.float64,
            value_type=numba.int64,
        )

        first_point = np.array([1.2, 1.4, 1.6], dtype=np.float64)
        second_point = np.array([1.3, 1.5, 1.7], dtype=np.float64)
        vertices = np.zeros((2, 3), dtype=np.float64)
        key1 = first_point[0] * 3 + first_point[1] * 2 + first_point[2]
        key2 = second_point[0] * 3 + second_point[1] * 2 + second_point[2]

        index = add_middle_get_index.py_func(
            first_point,
            vertices,
            dictionary,
        )
        index = add_middle_get_index.py_func(
            second_point,
            vertices,
            dictionary,
        )
        self.assertEqual(index, 1)
        self.assertEqual(dictionary[key2], 1)

    def test_add_same_point(self):
        dictionary = Dict.empty(
            key_type=numba.float64,
            value_type=numba.int64,
        )

        first_point = np.array([1.2, 1.4, 1.6], dtype=np.float64)
        vertices = np.zeros((2, 3), dtype=np.float64)
        key = first_point[0] * 3 + first_point[1] * 2 + first_point[2]

        index = add_middle_get_index.py_func(
            first_point,
            vertices,
            dictionary,
        )
        index = add_middle_get_index.py_func(
            first_point,
            vertices,
            dictionary,
        )
        self.assertEqual(index, 0)
        self.assertEqual(dictionary[key], 0)

    def test_middle_coords(self):
        middle_coords = get_middle_coords.py_func(
            np.array([0, 0, 0], dtype=np.float64), np.array(
                [2, 2, 2], dtype=np.float64)
        )
        self.assertEqual(middle_coords[0], 1)
        self.assertEqual(middle_coords[1], 1)
        self.assertEqual(middle_coords[2], 1)

    def test_list(self):
        result = get_middle_coords.py_func(
            np.array([2, 2, 2], dtype=np.float64), np.array(
                [0, 0, 0], dtype=np.float64)
        )
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 1)
        self.assertEqual(result[0], 1)
