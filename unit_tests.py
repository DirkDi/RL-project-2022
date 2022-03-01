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

    def test_weighted_map(self):
        """
        Checks if the weighted map is correctly created from the distance matrix, traffic matrix and traffic lights.
        """
        # TODO: think about tests
        pass

    def test_environment_creation(self):
        """
        Checks if the creation of the environment is allowed or not and if parameters work correctly.
        """
        # TODO: think about tests
        pass

    def test_action_step(self):
        """
        Checks if an action step inside the environment works correctly.
        """
        # TODO: think about tests (what should be reached after specific action? what should happen if package is at
        #  start point?)
        pass

    def test_reset_environment(self):
        """
        Checks if the reset of the environment works correctly.
        """
        # TODO: think about tests
        pass

    def test_get_max_emission_action(self):
        """
        Checks if the maximum weight/emission baseline works correct.
        """
        # TODO: think about tests (test one step, test all steps)
        pass

    def test_get_min_emission_action(self):
        """
        Checks if the minimum weight/emission baseline works correct.
        """
        # TODO: think about tests (test one step, test all steps)
        pass

    def test_stored_sarsa(self):
        """
        Checks if storing and loading of trained sarsa agent is correct.
        """
        # TODO: think about tests
        pass
