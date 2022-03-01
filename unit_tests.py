import unittest
import numpy as np
from env import CityEnv


class Test(unittest.TestCase):
    """
    This class is used to test the correctness of implementations (unittests).
    """
    def test_validate_accessibility(self):
        """
        Checks if the validation of accessibility is correct.
        """
        dist1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        dist2 = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        env = CityEnv(height=2, width=2, one_way=False, construction_sites=False, traffic_lights=False)
        env.dist_matrix = dist1
        # check for direct paths
        self.assertTrue(env.validate_accessibility(0, 0))  # start and end is the same so there always accessibility
        self.assertFalse(env.validate_accessibility(-1, 0))  # out of range (negative) has no accessibility
        self.assertFalse(env.validate_accessibility(0, -1))  # out of range (negative) has no accessibility
        self.assertFalse(env.validate_accessibility(-1, -1))  # out of range (negative) has no accessibility
        self.assertFalse(env.validate_accessibility(4, 0))  # out of range (positive) has no accessibility
        self.assertFalse(env.validate_accessibility(0, 4))  # out of range (positive) has no accessibility
        self.assertFalse(env.validate_accessibility(4, 4))  # out of range (positive) has no accessibility
        self.assertTrue(env.validate_accessibility(0, 1))  # Path exists (0 -> 1)
        self.assertTrue(env.validate_accessibility(1, 0))  # Path exists (1 -> 0)
        self.assertTrue(env.validate_accessibility(2, 3))  # Path exists (2 -> 3)
        self.assertFalse(env.validate_accessibility(3, 2))  # No Path exists because edge is unidirectional
        self.assertFalse(env.validate_accessibility(0, 2))  # No path exists
        # check for indirect paths
        env.dist_matrix = dist2
        self.assertTrue(env.validate_accessibility(0, 3))  # Path exists (0 -> 1 -> 3)
        self.assertFalse(env.validate_accessibility(3, 0))  # No Path exists because edges are unidirectional
        self.assertFalse(env.validate_accessibility(0, 2))  # No path exists

    def test_get_max_emission_action(self):
        """
        Checks if the maximum weight/emission baseline works correct.
        """
        pass

    def test_get_min_emission_action(self):
        """
        Checks if the minimum weight/emission baseline works correct.
        """
        pass

