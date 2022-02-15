import numpy as np
import gym

# Actions
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class CityEnv(gym.Env):

    def __init__(self, length=3, width=3, min_distance=10, max_distance=100, min_traffic=1, max_traffic=2,
                 dist_matrix: np.ndarray = None,
                 traffic_matrix: np.ndarray = None):
        self.length = length
        self.width = width
        self.min_distance = min_distance  # minimum distance between vertices
        self.max_distance = max_distance  # maximum distance between vertices
        self.min_traffic = min_traffic  # minimum traffic occurence between vertices
        self.max_traffic = max_traffic  # maximum traffic occurence between vertices
        self.matrix_length = self.length * self.width
        self.pos = (0, 0)
        self.vertice_matrix = np.reshape(np.arange(0, self.matrix_length), (-1, self.length))

        if dist_matrix is None:
            dist_matrix = np.zeros((self.matrix_length, self.matrix_length))

        for i in range(self.matrix_length):
            for j in range(self.matrix_length):
                if j == i + self.length or j == i - self.length:
                    dist_matrix[j][i] = 1
                if j == i+1 and j % self.length != 0:
                    print(i, j)
                    dist_matrix[j][i] = 1
                if j == i-1 and i % self.length != 0:
                    dist_matrix[j][i] = 1

        if traffic_matrix is None:
            traffic_matrix = np.ones((self.matrix_length, self.matrix_length))

        for k in range(self.matrix_length):
            dist_matrix[k][k] = 0
            traffic_matrix[k][k] = 0

        self.dist_matrix = dist_matrix.copy()


env = CityEnv()

for row in env.dist_matrix:
    print(row)

