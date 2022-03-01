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
        dist = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        env = CityEnv(height=2, width=2, one_way=False, construction_sites=False, traffic_lights=False)
        env.dist_matrix = dist
        self.assertTrue(env.validate_accessibility(0, 1))
        self.assertFalse(env.validate_accessibility(0, 2))

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

