import unittest
import numpy as np 
import pandas as pd
import math
import datetime
import os

from LinearRegression import LinearRegression

class LinearRegressionTest(unittest.TestCase):

    arr_1 = [1, 2, 3, 4]
    arr_2 = [5, 6, 7]

    # Invalid input length.
    def test_mismatched_lengths(self):
        with self.assertRaises(Exception):
            LinearRegression(arr_1, arr_2)

    arr_1 = np.array([1, 2, 3, 4]).reshape(2, 2)
    arr_2 = np.array([4, 5, 2, 3])

    # Invalid input shape.
    def test_mismatched_shapes(self):
        with self.assertRaises(Exception):
            LinearRegression(arr_1, arr_2)

    arr_1 = 'Hello'
    arr_2 = 2

    # Invalid input type.
    def test_invalid_parameter_type(self):
        with self.assertRaises(Exception):
            LinearRegression(arr_1, arr_2)

    arr_1 = 2
    arr_2 = [1, 2, 3, 4]
    
    # Invalid first input type.
    def test_invalid_parameter_one_type(self):
        with self.assertRaises(Exception):
            LinearRegression(arr_1, arr_2)
    
    arr_1 = [1, 2, 3, 4]
    arr_2 = 2
    
    # Invalid second input type.
    def test_invalid_parameter_two_type(self):
        with self.assertRaises(Exception):
            LinearRegression(arr_1, arr_2)

if __name__ == '__main__':
    unittest.main()