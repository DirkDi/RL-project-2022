import numpy as np
import gym
import sys
import logging
import random

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class CityEnv(gym.Env):
    """
    An environment to simulate a city traffic
    """

    def __init__(self, length=3, width=3, min_distance=10, max_distance=100, min_traffic=1, max_traffic=2,
                 dist_matrix: np.ndarray = None,
                 traffic_matrix: np.ndarray = None,
                 packages=None,
                 num_packages: int = 2, init_random=False):
        """
        Initialize the environment
        """
        # throw error message if environment is not possible
        if any(value <= 0 for value in [length, width, min_distance, max_distance, min_traffic, max_traffic,
                                        num_packages]) or max_distance < min_distance or max_traffic < min_traffic:
            logging.error('This environment is not possible.')
            sys.exit(1)

        self.length = length
        self.width = width
        self.min_distance = min_distance  # minimum distance between vertices
        self.max_distance = max_distance  # maximum distance between vertices
        self.min_traffic = min_traffic  # minimum traffic occurrence between vertices
        self.max_traffic = max_traffic  # maximum traffic occurrence between vertices
        self.matrix_length = self.length * self.width
        self.pos = 0, 0
        self.prev_pos = 0, 0
        self.vertices_matrix = np.reshape(np.arange(0, self.matrix_length), (-1, self.length))

        if dist_matrix is None:
            dist_matrix = np.zeros((self.matrix_length, self.matrix_length))
            for i in range(self.matrix_length):
                for j in range(self.matrix_length):
                    if dist_matrix[j][i] != 0:
                        pass
                    elif j == i + self.length or (j == i + 1 and j % self.length != 0):
                        rand_val = random.randint(min_distance, max_distance)
                        dist_matrix[j][i] = rand_val if init_random else 1
                        dist_matrix[i][j] = rand_val if init_random else 1
        logging.debug(f'Distance matrix:\n{dist_matrix}')
        # create values for traffic
        if traffic_matrix is None:
            traffic_matrix = np.zeros((self.matrix_length, self.matrix_length))
            for i in range(self.matrix_length):
                for j in range(self.matrix_length):
                    if traffic_matrix[j][i] != 0:
                        pass
                    elif j == i + self.length or (j == i + 1 and j % self.length != 0):
                        rand_val = round(random.uniform(min_traffic, max_traffic), 2)
                        traffic_matrix[j][i] = rand_val if init_random else 1
                        traffic_matrix[i][j] = rand_val if init_random else 1
        logging.debug(f'Traffic matrix:\n{traffic_matrix}')

        if packages is None:
            packages = []
            for i in range(num_packages):
                packages.append((random.randint(0, self.length - 1), random.randint(0, self.width - 1)))
            #packages.append((0, 1))
            #packages.append((2, 1))
        logging.debug(f'Coordinates of packages are: {packages}')

        self.dist_matrix = dist_matrix.copy()
        self.traffic_matrix = traffic_matrix.copy()
        self.packages = packages.copy()
        self.packages_initial = packages.copy()

        low = np.array([0, 0])
        high = np.array([self.length, self.width])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint32)
        self.action_space = gym.spaces.Discrete(4)
        # TODO: define reward range
        self.reward_range = [0, 0]

    def reset(self):
        """
        Reset the environment
        """
        self.pos = 0, 0
        self.packages = self.packages_initial.copy()

    def step(self, action):
        """
        Performs a step on the environment
        """
        action = int(action)
        if action < 0 or action >= 4:
            raise RuntimeError(f"{action} is not a valid action (needs to be between 0 and 3)")
        i, j = self.pos

        x, y = i, j
        if action == 0 and i > 0:
            x = i - 1
        elif action == 1 and i < self.length - 1:
            x = i + 1
        elif action == 2 and j > 0:
            y = j - 1
        elif action == 3 and j < self.width - 1:
            y = j + 1
        else:
            return self.pos, -10000, False, {}
        old_vertex = self.vertices_matrix[i, j]
        new_vertex = self.vertices_matrix[x, y]
        dist = self.dist_matrix[new_vertex, old_vertex]
        traffic_flow = self.traffic_matrix[new_vertex, old_vertex]
        # print(old_vertex, new_vertex, dist, traffic_flow)
        self.pos = x, y
        m, n = self.packages[0]
        if (x, y) == (m, n):
            while (x, y) in self.packages:
                self.packages.remove((x, y))
            dist_to_package = 0
        else:
            dist_to_package = abs(m - x) + abs(n - y)

        reward = 10 * (dist * traffic_flow + 2 * dist_to_package)  # TODO: check if reward has good value
        done = len(self.packages) == 0
        meta_info = {}
        # print(1 / reward * 1000, self.pos)
        # sys.exit(0)
        logging.debug(f'Current packages: {self.packages}; new position after step is {self.pos}')
        return self.pos, -reward, done, meta_info

    def close(self):
        """
        Make sure environment is closed
        """
        pass
