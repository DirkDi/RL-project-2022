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
        Checks if the weighted map is correctly created from the distance matrix and traffic matrix.
        """
        # create different distance and traffic matrices to check edge cases
        dist1 = np.ones((9, 9)) * 10
        traffic1 = np.ones((9, 9)) * 1.5
        correct_map1 = np.ones((9, 9)) * 15
        env1 = CityEnv(height=3, width=3, dist_matrix=dist1, traffic_matrix=traffic1, one_way=False,
                       construction_sites=False, traffic_lights=False)
        dist2 = np.zeros((9, 9))
        traffic2 = np.ones((9, 9))
        correct_map2 = np.zeros((9, 9))
        env2 = CityEnv(height=3, width=3, dist_matrix=dist2, traffic_matrix=traffic2, one_way=False,
                       construction_sites=False, traffic_lights=False)
        dist3 = np.ones((9, 9))
        traffic3 = np.zeros((9, 9))
        correct_map3 = np.zeros((9, 9))
        env3 = CityEnv(height=3, width=3, dist_matrix=dist3, traffic_matrix=traffic3, one_way=False,
                       construction_sites=False, traffic_lights=False)
        dist4 = np.zeros((9, 9))
        traffic4 = np.zeros((9, 9))
        correct_map4 = np.zeros((9, 9))
        env4 = CityEnv(height=3, width=3, dist_matrix=dist4, traffic_matrix=traffic4, one_way=False,
                       construction_sites=False, traffic_lights=False)
        # test the environments
        self.assertEqual(env1.weighted_map, correct_map1)
        self.assertEqual(env1.get_map(), correct_map1)
        self.assertEqual(env2.weighted_map, correct_map2)
        self.assertEqual(env2.get_map(), correct_map2)
        self.assertEqual(env3.weighted_map, correct_map3)
        self.assertEqual(env3.get_map(), correct_map3)
        self.assertEqual(env4.weighted_map, correct_map4)
        self.assertEqual(env4.get_map(), correct_map4)

    def test_environment_creation(self):
        """
        Checks if the creation of the environment is allowed or not and if parameters work correctly.
        """
        # check if assertion is raised if one of the variables "height", "width", "min_distance", "max_distance",
        #                                 "min_traffic", "max_traffic", "num_packages" is <= 0.
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=-1, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=-1, width=3, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=-1, width=-1, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=0, width=0, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=3, min_distance=0, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=3, min_distance=0, max_distance=0, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=3, min_traffic=0, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=3, min_traffic=0, max_traffic=0, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=3, num_packages=0, one_way=False,
                    construction_sites=False, traffic_lights=False)
        # check if assertion is raised if "max_distance" is < "min_distance" or "max_traffic" < "min_traffic".
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=3, min_distance=5, max_distance=4, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=3, width=3, min_traffic=5, max_traffic=4, one_way=False,
                    construction_sites=False, traffic_lights=False)
        # check if the variable "num_packages" is correct after giving default packages
        env = CityEnv(height=3, width=3, packages=[(0, 1), (1, 0), (1, 1), (2, 0)], one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual(env.num_packages, 4)
        self.assertEqual(len(env.packages), 4)
        env = CityEnv(height=3, width=3, packages=[(0, 1)], one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual(env.num_packages, 1)
        self.assertEqual(len(env.packages), 1)
        env = CityEnv(height=3, width=3, num_packages=100, packages=[(0, 1), (1, 0)], one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual(env.num_packages, 2)
        self.assertEqual(len(env.packages), 2)
        env = CityEnv(height=3, width=3, num_packages=4, packages=[], one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual(env.num_packages, 4)
        self.assertEqual(len(env.packages), 4)
        env = CityEnv(height=3, width=3, one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual(env.num_packages, 2)
        self.assertEqual(len(env.packages), 2)

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
