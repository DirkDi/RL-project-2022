import numpy as np
import gym

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
                 num_packages: int = 1):
        """
        Initialize the environment
        """
        self.length = length
        self.width = width
        self.min_distance = min_distance  # minimum distance between vertices
        self.max_distance = max_distance  # maximum distance between vertices
        self.min_traffic = min_traffic  # minimum traffic occurrence between vertices
        self.max_traffic = max_traffic  # maximum traffic occurrence between vertices
        self.matrix_length = self.length * self.width
        self.pos = 0, 0
        self.vertices_matrix = np.reshape(np.arange(0, self.matrix_length), (-1, self.length))

        if dist_matrix is None:
            dist_matrix = np.zeros((self.matrix_length, self.matrix_length))

            for i in range(self.matrix_length):
                for j in range(self.matrix_length):
                    if j == i + self.length or j == i - self.length:
                        dist_matrix[j][i] = 1
                    if j == i+1 and j % self.length != 0:
                        # print(i, j)
                        dist_matrix[j][i] = 1
                    if j == i-1 and i % self.length != 0:
                        dist_matrix[j][i] = 1

        if traffic_matrix is None:
            traffic_matrix = np.ones((self.matrix_length, self.matrix_length))

            for i in range(self.matrix_length):
                for j in range(self.matrix_length):
                    if j == i + self.length or j == i - self.length:
                        traffic_matrix[j][i] = 1
                    if j == i + 1 and j % self.length != 0:
                        # print(i, j)
                        traffic_matrix[j][i] = 1
                    if j == i - 1 and i % self.length != 0:
                        traffic_matrix[j][i] = 1

            for k in range(self.matrix_length):
                # dist_matrix[k][k] = 0
                traffic_matrix[k][k] = 0

        if packages is None:
            packages = []
            for i in range(num_packages):
                packages.append((2, 1))

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
            return self.pos, 0, False, {}
        old_vertex = self.vertices_matrix[i, j]
        new_vertex = self.vertices_matrix[x, y]
        dist = self.dist_matrix[new_vertex, old_vertex]
        traffic_flow = self.traffic_matrix[new_vertex, old_vertex]
        # print(old_vertex, new_vertex, dist, traffic_flow)
        self.pos = i, j
        m, n = self.packages[0]
        if (x, y) == (m, n):
            self.packages.pop()
            dist_to_package = 0
        else:
            dist_to_package = abs(m - x) + abs(n - y)
        reward = dist * traffic_flow + 0.1 * dist_to_package
        done = len(self.packages) == 0
        meta_info = {}
        return self.pos, 1 / reward * 1000, done, meta_info

    def close(self):
        """
        Make sure environment is closed
        """
        pass
