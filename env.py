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

    def __init__(self, height=3, width=3, min_distance=10, max_distance=100, min_traffic=1, max_traffic=2,
                 dist_matrix: np.ndarray = None,
                 traffic_matrix: np.ndarray = None,
                 packages=None,
                 num_packages: int = 2, init_random=False):
        """
        Initialize the environment
        """
        # throw error message if environment is not possible
        if any(value <= 0 for value in [height, width, min_distance, max_distance, min_traffic, max_traffic,
                                        num_packages]) or max_distance < min_distance or max_traffic < min_traffic:
            logging.error('Environment out of range.')
            sys.exit(1)

        self.height = height
        self.width = width
        self.min_distance = min_distance  # minimum distance between vertices
        self.max_distance = max_distance  # maximum distance between vertices
        self.min_traffic = min_traffic  # minimum traffic occurrence between vertices
        self.max_traffic = max_traffic  # maximum traffic occurrence between vertices
        self.matrix_height = self.height * self.width
        self.pos = 0, 0
        self.prev_pos = 0, 0
        self.vertices_matrix = np.reshape(np.arange(0, self.matrix_height), (-1, self.height))
        self.timer = 0

        if dist_matrix is None:
            dist_matrix = np.zeros((self.matrix_height, self.matrix_height))
            for i in range(self.matrix_height):
                for j in range(self.matrix_height):
                    if dist_matrix[j][i] != 0:
                        pass
                    elif j == i + self.height or (j == i + 1 and j % self.height != 0):
                        rand_val = random.randint(min_distance, max_distance)
                        dist_matrix[j, i] = rand_val if init_random else 1
                        dist_matrix[i, j] = rand_val if init_random else 1
        logging.debug(f'Distance matrix:\n{dist_matrix}')
        # create values for traffic
        if traffic_matrix is None:
            traffic_matrix = np.zeros((self.matrix_height, self.matrix_height))
            for i in range(self.matrix_height):
                for j in range(self.matrix_height):
                    if traffic_matrix[j][i] != 0:
                        pass
                    elif j == i + self.height or (j == i + 1 and j % self.height != 0):
                        rand_val = round(random.uniform(min_traffic, max_traffic), 2)
                        traffic_matrix[j, i] = rand_val if init_random else 1
                        traffic_matrix[i, j] = rand_val if init_random else 1
        logging.debug(f'Traffic matrix:\n{traffic_matrix}')

        if packages is None:
            packages = []
            if init_random:
                for i in range(num_packages):
                    packages.append((random.randint(0, self.height - 1), random.randint(0, self.width - 1)))
            else:
                packages.append((2, 1))
        logging.debug(f'Coordinates of packages are: {packages}')

        self.dist_matrix = dist_matrix.copy()
        self.traffic_matrix = traffic_matrix.copy()
        self.packages = packages.copy()
        self.packages_initial = packages.copy()

        low = np.array([0, 0])
        high = np.array([self.height, self.width])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint32)
        self.action_space = gym.spaces.Discrete(4)
        # TODO: define reward range
        self.reward_range = [0, 0]

    def reset(self):
        """
        Reset the environment
        """
        self.timer = 0
        self.pos = 0, 0
        self.packages = self.packages_initial.copy()
        return self.pos, len(self.packages)

    def step(self, action):
        """
        Performs a step on the environment
        """
        action = int(action)
        if action < 0 or action >= 4:
            raise RuntimeError(f"{action} is not a valid action (needs to be between 0 and 3)")

        self.timer += 1
        pos_x, pos_y = self.pos
        new_pos_x, new_pos_y = pos_x, pos_y
        if action == UP and pos_x > 0:
            new_pos_x -= 1
        elif action == DOWN and pos_x < self.height - 1:
            new_pos_x += 1
        elif action == LEFT and pos_y > 0:
            new_pos_y -= 1
        elif action == RIGHT and pos_y < self.width - 1:
            new_pos_y += 1
        else:
            dist_to_next_package = self.height * self.width
            reward = 0
            for pack_x, pack_y in self.packages:
                dist_to_next_package = min(dist_to_next_package, abs(pack_x - pos_x) + abs(pack_y - pos_y))
                reward = 1 / dist_to_next_package
            return (self.pos, len(self.packages)), reward - self.timer, False, {}

        self.pos = new_pos_x, new_pos_y
        start_vertex = self.vertices_matrix[pos_x, pos_y]
        target_vertex = self.vertices_matrix[new_pos_x, new_pos_y]
        dist = self.dist_matrix[start_vertex, target_vertex]
        traffic_flow = self.traffic_matrix[start_vertex, target_vertex]
        if (new_pos_x, new_pos_y) in self.packages:
            while (new_pos_x, new_pos_y) in self.packages:
                self.packages.remove((new_pos_x, new_pos_y))

        packages_count = len(self.packages)
        done = packages_count == 0
        if done:
            reward = 2
        else:
            dist_to_next_package = self.height * self.width
            for pack_x, pack_y in self.packages:
                dist_to_next_package = min(dist_to_next_package, abs(pack_x - new_pos_x) + abs(pack_y - new_pos_y))
            reward = 50 * (1 / (dist * traffic_flow + 10 * dist_to_next_package))
        meta_info = {}
        return (self.pos, packages_count), reward - self.timer, done, meta_info

    def close(self):
        """
        Make sure environment is closed
        """
        pass
