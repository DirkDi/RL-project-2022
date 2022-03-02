import unittest
import numpy as np
from env import CityEnv


class Test(unittest.TestCase):
    """
    This class is used to test the correctness of implementations (unittests).
    """

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
        # check if assertion is raised if "distance_matrix" or "traffic_matrix" contains negative values.
        neg_dist = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0]
        ])
        neg_traffic = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        pos_dist = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        pos_traffic = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        with self.assertRaises(AssertionError):
            CityEnv(height=2, width=2, dist_matrix=neg_dist, traffic_matrix=neg_traffic, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=2, width=2, dist_matrix=neg_dist, traffic_matrix=pos_traffic, one_way=False,
                    construction_sites=False, traffic_lights=False)
        with self.assertRaises(AssertionError):
            CityEnv(height=2, width=2, dist_matrix=pos_dist, traffic_matrix=neg_traffic, one_way=False,
                    construction_sites=False, traffic_lights=False)
        # check if assertion is raised if "dist_matrix" and "traffic_matrix" have not the same dimension
        with self.assertRaises(AssertionError):
            dist = np.ones((2, 2))
            traffic = np.ones((3, 3))
            CityEnv(dist_matrix=dist, traffic_matrix=traffic, one_way=False,
                    construction_sites=False, traffic_lights=False)
        # check if assertion is raised if "dist_matrix" and "traffic_matrix" have not the same edges
        with self.assertRaises(AssertionError):
            dist = np.ones((2, 2))
            traffic = np.zeros((2, 2))
            CityEnv(dist_matrix=dist, traffic_matrix=traffic, one_way=False,
                    construction_sites=False, traffic_lights=False)
        # check if "height" and "width" values are correct after initialization
        env = CityEnv(height=2, width=2, one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual((env.height, env.width), (2, 2))
        env = CityEnv(height=5, width=5, one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual((env.height, env.width), (5, 5))
        env = CityEnv(one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual((env.height, env.width), (3, 3))
        dist = np.ones((3, 3))
        traffic = np.ones((3, 3))
        env = CityEnv(height=1, width=1, dist_matrix=dist, traffic_matrix=traffic, one_way=False,
                      construction_sites=False, traffic_lights=False)
        self.assertEqual((env.height, env.width), (3, 3))
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

    def test_reset_environment(self):
        """
        Checks if the reset of the environment works correctly.
        """
        env = CityEnv(height=3, width=3, num_packages=4, one_way=False,
                      construction_sites=False, traffic_lights=False)
        # get old packages to see if reset can load them.
        old_packages = env.packages.copy()
        # change variables to check if reset works correctly.
        env.pos = (2, 2)
        env.init_pos = (0, 0)
        env.packages = []
        # reset environment to test the initial position
        init_values = env.reset()
        # test if position is the initial position
        self.assertEqual(env.pos, (init_values[0], init_values[1]))
        self.assertEqual(env.pos, (0, 0))
        self.assertEqual((init_values[0], init_values[1]), (0, 0))
        # test if amount of packages is 4 (start amount)
        self.assertEqual(init_values[2], len(env.packages))
        self.assertEqual(init_values[2], 4)
        self.assertEqual(len(env.packages), 4)
        # test if the newly initialized packages are the same as previous
        self.assertEqual(env.packages, old_packages)

    def test_action_step(self):
        """
        Checks if an action step inside the environment works correctly.
        """
        # create environment to make actions
        dist = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        traffic = np.array([
            [0, 5, 5, 0],
            [5, 0, 0, 5],
            [5, 0, 0, 5],
            [0, 5, 5, 0]
        ])
        env = CityEnv(height=2, width=2, packages=[(0, 1), (1, 1)], dist_matrix=dist, traffic_matrix=traffic,
                      one_way=False, construction_sites=False, traffic_lights=False)
        env.init_pos = (0, 0)
        # actions are defined from 0 to 3
        with self.assertRaises(RuntimeError):
            env.step(-1)
        with self.assertRaises(RuntimeError):
            env.step(4)
        env.reset()
        # test if unallowed directions are heavily penalized (-1000) and if position won't change after this action
        _, reward, done, _ = env.step(0)
        self.assertEqual(reward, -1000)
        self.assertEqual(env.pos, (0, 0))
        self.assertFalse(done)
        env.reset()
        _, reward, done, _ = env.step(2)
        self.assertEqual(reward, -1000)
        self.assertEqual(env.pos, (0, 0))
        self.assertFalse(done)
        # test if rewards are correct for allowed steps/actions
        env.reset()
        _, reward, done, _ = env.step(3)
        self.assertEqual(reward, -5)
        self.assertEqual(env.pos, (0, 1))
        self.assertFalse(done)
        _, reward, done, _ = env.step(1)
        self.assertEqual(reward, -5)
        self.assertEqual(env.pos, (1, 1))
        self.assertTrue(done)
        # test reward of traffic light (add traffic light to environment)
        env.reset()
        env.traffic_lights = [(0, 1)]
        _, reward, _, _ = env.step(3)
        self.assertEqual(reward, -6)
        _, reward, _, _ = env.step(2)
        self.assertEqual(reward, -5)

    def test_get_max_emission_action(self):
        """
        Checks if the maximum weight/emission baseline works correct.
        """
        # TODO: think about tests (test one step, test all steps)
        dist = np.array([
            [0, 1, 100, 0],
            [1, 0, 0, 1],
            [100, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        traffic = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        env = CityEnv(dist_matrix=dist, traffic_matrix=traffic, packages=[(1, 0)],
                      one_way=False, construction_sites=False, traffic_lights=False)
        env.init_pos = 0, 0
        env.reset()
        self.assertEqual(env.get_max_emission_action(), 1)

    def test_get_min_emission_action(self):
        """
        Checks if the minimum weight/emission baseline works correct.
        """
        # TODO: think about tests (test one step, test all steps)
        dist = np.array([
            [0, 1, 100, 0],
            [1, 0, 0, 1],
            [100, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        traffic = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        env = CityEnv(dist_matrix=dist, traffic_matrix=traffic, packages=[(1, 0)],
                      one_way=False, construction_sites=False, traffic_lights=False)
        env.init_pos = 0, 0
        env.reset()
        self.assertEqual(env.get_min_emission_action(), 3)

    def test_stored_sarsa(self):
        """
        Checks if storing and loading of trained sarsa agent is correct.
        """
        # TODO: think about tests
        pass
