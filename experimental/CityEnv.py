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
    def __init__(self, height=3, width=3,
                 min_distance=1, max_distance=100,
                 min_traffic=1, max_traffic=2,
                 dist_matrix=None, traffic_matrix=None,
                 packages=None, num_packages=1,
                 init_random=False):
        """
        Initialize the environment
        """
        self.height = height
        self.width = width
        self.min_distance = min_distance  # minimum distance between vertices
        self.max_distance = max_distance  # maximum distance between vertices
        self.min_traffic = min_traffic    # minimum traffic occurrence between vertices
        self.max_traffic = max_traffic    # maximum traffic occurrence between vertices

        matrix_size = height * width
        self.vertices_matrix = np.reshape(np.arange(0, matrix_size), (-1, height))

        if dist_matrix is None:
            dist_matrix = np.zeros((matrix_size, matrix_size))
            for i in range(height):
                for j in range(width):
                    start_vertex = self.vertices_matrix[i, j]
                    for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= a < height and 0 <= b < width:
                            target_vertex = self.vertices_matrix[a, b]
                            dist_matrix[start_vertex, target_vertex] = 1

        if traffic_matrix is None:
            traffic_matrix = np.zeros((matrix_size, matrix_size))
            for i in range(height):
                for j in range(width):
                    start_vertex = self.vertices_matrix[i, j]
                    for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= a < height and 0 <= b < width:
                            target_vertex = self.vertices_matrix[a, b]
                            traffic_matrix[start_vertex, target_vertex] = 1

        if packages is None:
            packages = []
            packages.append((2, 1))

        # Indexing scheme: matrix[start_vertex, target_vertex]
        self.dist_matrix = dist_matrix
        self.traffic_matrix = traffic_matrix
        self.packages = []
        self.packages_initial = packages
        self.position = 0, 0
        self.timer = 0

        # state := ((x, y), amount of not collected packages)
        low = np.array(
            [
                0,  # top border
                0,  # left border
                0   # low count of packages
            ]
        )
        high = np.array(
            [
                height,  # bottom border
                width,   # right border
                100      # TODO: redefine high count of packages
            ]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint32)
        self.action_space = gym.spaces.Discrete(4)
        # TODO: define reward range
        self.reward_range = [0, 0]

    def reset(self):
        """
        Reset the environment
        """
        self.position = 0, 0
        self.timer = 0
        self.packages = self.packages_initial.copy()
        return self.position, len(self.packages)

    def step(self, action):
        """
        Performs a step on the environment
        """
        action = int(action)
        if action < 0 or action >= 4:
            raise RuntimeError(f"{action} is not a valid action (needs to be between 0 and 3)")

        self.timer += 1
        pos_x, pos_y = self.position
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
            return (self.position, len(self.packages)), reward - self.timer, False, {}

        self.position = new_pos_x, new_pos_y
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
            reward = 1 / dist_to_next_package
        meta_info = {}
        return (self.position, packages_count), reward - self.timer, done, meta_info

    def close(self):
        """
        Make sure environment is closed
        """
        pass
