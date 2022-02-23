import numpy as np
import gym
import sys
import logging
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import pylab
from collections import Counter

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
                 num_packages: int = 2, init_random=False, one_way=True, construction_sites=True,
                 traffic_lights=True):
        """
        Initialize the environment
        """
        super(CityEnv, self).__init__()
        # throw error message if environment is not possible
        assert np.all(np.array([height, width, min_distance, max_distance,
                                min_traffic, max_traffic, num_packages]) > 0), "all arguments must be non-negative!"
        assert min_distance < max_distance and min_traffic < max_traffic, \
            "minimum values have to be lower than maximum values!"
        """
        if any(value <= 0 for value in [height, width, min_distance, max_distance, min_traffic, max_traffic,
                                        num_packages]) or max_distance < min_distance or max_traffic < min_traffic:
            logging.error('Environment out of range.')
            sys.exit(1)
        """
        self.height = height
        self.width = width
        self.min_distance = min_distance  # minimum distance between vertices
        self.max_distance = max_distance  # maximum distance between vertices
        self.min_traffic = min_traffic  # minimum traffic occurrence between vertices
        self.max_traffic = max_traffic  # maximum traffic occurrence between vertices

        # Create vertices matrix
        self.matrix_height = self.height * self.width
        self.vertices_matrix = np.reshape(np.arange(0, self.matrix_height), (-1, self.width))

        if dist_matrix is None:
            dist_matrix = np.zeros((self.matrix_height, self.matrix_height))
            for i in range(self.height):
                for j in range(self.width):
                    start_vertex = self.vertices_matrix[i, j]
                    for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= a < self.height and 0 <= b < self.width:
                            target_vertex = self.vertices_matrix[a, b]
                            if dist_matrix[start_vertex, target_vertex] > 0:
                                continue
                            dist = random.randint(min_distance, max_distance) if init_random else 1
                            dist_matrix[start_vertex, target_vertex] = dist
                            dist_matrix[target_vertex, start_vertex] = dist
        print("dist_matrix created")
        # create values for traffic
        if traffic_matrix is None:
            traffic_matrix = np.zeros((self.matrix_height, self.matrix_height))
            for i in range(self.height):
                for j in range(self.width):
                    start_vertex = self.vertices_matrix[i, j]
                    for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= a < self.height and 0 <= b < self.width:
                            target_vertex = self.vertices_matrix[a, b]
                            if traffic_matrix[start_vertex, target_vertex] > 0:
                                continue
                            flow = round(random.uniform(min_traffic, max_traffic), 2) if init_random else 1
                            traffic_matrix[start_vertex, target_vertex] = flow
                            traffic_matrix[target_vertex, start_vertex] = flow
        print("traffic_matrix created")
        self.dist_matrix = dist_matrix.copy()
        self.traffic_matrix = traffic_matrix.copy()
        # Check if city graph is connected
        for i in range(self.matrix_height):
            for j in range(self.matrix_height):
                assert self.validate_accessibility(i, j), "The city graph is not connected!"

        self.init_pos = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
        self.pos = self.init_pos
        self.prev_pos = self.init_pos
        logging.debug(f'The start position is {self.init_pos}')

        self.timer = 0
        self.num_packages = num_packages
        minimal_generating = min(self.height - 1, self.width - 1)
        if minimal_generating:
            self.num_one_way = random.randint(1, min(self.height - 1, self.width - 1))
            self.num_construction_sites = random.randint(1, min(self.height - 1, self.width - 1))
            self.num_traffic_lights = random.randint(1, min(self.height - 1, self.width - 1))
        else:
            self.num_one_way = 0
            self.num_construction_sites = 0
            self.num_traffic_lights = 0
        self.already_driven = [self.pos]  # contains the points where the agent already was
        self.traffic_lights = []  # list of coordinates with traffic lights
        self.dist = 0

        if packages is None:
            packages = []
            if init_random:
                for i in range(num_packages):
                    packages.append((random.randint(0, self.height - 1), random.randint(0, self.width - 1)))
            else:
                packages.append((2, 1))
        logging.debug(f'Coordinates of packages are: {packages}')
        self.packages = packages.copy()
        self.packages_initial = packages.copy()
        self.weighted_map = self.get_map()
        print("weighted_map created")
        if one_way:
            self.generate_one_way_streets()
        print("one way streets created")
        if construction_sites:
            self.generate_construction_sites()
        print("construction sites created")
        if traffic_lights:
            self.generate_traffic_lights()
        logging.debug(f'Distance matrix:\n{dist_matrix}')
        logging.debug(f'Traffic matrix:\n{traffic_matrix}')
        logging.debug(f'Weighted map matrix:\n{self.weighted_map}')
        low = np.array([0, 0, 0])
        high = np.array([self.height, self.width, num_packages])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.reward_range = [0, 0]  # TODO: define reward range
        self.already_driven = [(0, 0)]

    def reset(self):
        """
        Reset the environment
        """
        self.timer = 0
        self.pos = self.init_pos
        self.packages = self.packages_initial.copy()
        self.dist = 0
        return np.array([self.pos[0], self.pos[1], len(self.packages)]).astype(dtype=np.float32)

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
            """
            dist_to_next_package = self.height * self.width
            reward = 0
            for pack_x, pack_y in self.packages:
                dist_to_next_package = min(dist_to_next_package, abs(pack_x - pos_x) + abs(pack_y - pos_y))
                reward = 1 / dist_to_next_package
            """
            reward = 0  # -10000 * (self.height * self.width)
            return np.array([pos_x, pos_y, len(self.packages)]).astype(np.float32), reward, False, {
                'render.modes': ['console']}
        start_vertex = self.vertices_matrix[pos_x, pos_y]
        target_vertex = self.vertices_matrix[new_pos_x, new_pos_y]
        dist = self.dist_matrix[target_vertex, start_vertex]
        traffic_flow = self.traffic_matrix[target_vertex, start_vertex]
        # action is not allowed if there is no vertex between both points (value is 0 for dist & traffic_flow)
        if not dist:
            reward = 0  # -10000 * (self.height * self.width)
            return np.array([pos_x, pos_y, len(self.packages)]).astype(np.float32), reward, False, {
                'render.modes': ['console']}

        self.pos = new_pos_x, new_pos_y
        """
        if self.pos in self.already_driven:
            self.already_driven.append(self.pos)
            # count = Counter(self.already_driven)[self.pos]
            return np.array([new_pos_x, new_pos_y, len(self.packages)]).astype(np.float32), -10000 * (
                    self.height * self.width), False, {'render.modes': ['console']}
        """
        complete_dist = dist * traffic_flow if self.pos in self.traffic_lights else 1.2 * dist * traffic_flow
        self.dist += complete_dist
        reward = -(dist * traffic_flow + self.dist / 100) if self.pos not in self.traffic_lights else -(
                    dist * traffic_flow + self.dist / 100) * 1.2
        # reward = (1 / (dist * traffic_flow)) * 1000
        # reward = 1 / (dist * traffic_flow + self.dist / 10)
        if (new_pos_x, new_pos_y) in self.packages:
            while (new_pos_x, new_pos_y) in self.packages:
                self.packages.remove((new_pos_x, new_pos_y))
            self.already_driven = []  # reset already driven array
            # reward += 10 * (self.height + self.width + 1)

        packages_count = len(self.packages)
        done = packages_count == 0
        """
        if done:
            reward = 2
        else:
            dist_to_next_package = self.height * self.width
            for pack_x, pack_y in self.packages:
                dist_to_next_package = min(dist_to_next_package, abs(pack_x - new_pos_x) + abs(pack_y - new_pos_y))
            reward = 50 * (1 / (dist * traffic_flow + 10 * dist_to_next_package))
        """
        meta_info = {'render.modes': ['console']}
        self.already_driven.append((new_pos_x, new_pos_y))
        return np.array([new_pos_x, new_pos_y, packages_count]).astype(np.float32), reward, done, meta_info

    def close(self):
        """
        Make sure environment is closed
        """
        pass

    def render(self, mode="human"):
        """
        Render the environment
        """
        pass

    def generate_traffic_lights(self):
        print(self.num_traffic_lights)
        for i in range(self.num_traffic_lights):
            self.traffic_lights.append((random.randint(0, self.height - 1), random.randint(0, self.width - 1)))
        logging.debug(f'Traffic lights:\n {self.traffic_lights}, amount: {self.num_traffic_lights}')

    def generate_one_way_streets(self):
        used_points = []
        for i in range(self.num_one_way):
            taken_points = True
            start_vertex, target_vertex = -1, -1
            no_path = True
            while no_path:
                while taken_points:
                    start_vertex = self.vertices_matrix[
                        random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
                    target_idx = np.where(self.dist_matrix[:, start_vertex] > 0)[0]  # possible edges for one way
                    target_vertex = target_idx[random.randint(0, len(target_idx) - 1)]
                    if start_vertex not in used_points and target_vertex not in used_points:
                        taken_points = False
                old_dist = self.dist_matrix[start_vertex, target_vertex]
                old_traffic = self.traffic_matrix[start_vertex, target_vertex]
                old_weight = self.weighted_map[start_vertex, target_vertex]
                self.dist_matrix[start_vertex, target_vertex] = 0
                self.traffic_matrix[start_vertex, target_vertex] = 0
                self.weighted_map[start_vertex, target_vertex] = 0
                no_path = not self.validate_accessibility(start_vertex, target_vertex)
                if no_path:
                    self.dist_matrix[start_vertex, target_vertex] = old_dist
                    self.traffic_matrix[start_vertex, target_vertex] = old_traffic
                    self.weighted_map[start_vertex, target_vertex] = old_weight

            used_points += [start_vertex, target_vertex]
        logging.debug(f'One way streets:\n {used_points}, amount: {self.num_one_way}')

    def generate_construction_sites(self):
        used_points = []
        amount_points = self.height * self.width
        for i in range(self.num_construction_sites):
            taken_points = True
            start_vertex, target_vertex = -1, -1
            no_path = True
            counter = 0
            while no_path and counter <= amount_points:
                while taken_points:
                    start_vertex = self.vertices_matrix[
                        random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
                    target_idx = np.where(self.dist_matrix[:, start_vertex] > 0)[0]  # possible edges for one way
                    target_vertex = target_idx[random.randint(0, len(target_idx) - 1)]
                    if start_vertex not in used_points and target_vertex not in used_points:
                        taken_points = False
                old_dist = self.dist_matrix[start_vertex, target_vertex]
                old_traffic = self.traffic_matrix[start_vertex, target_vertex]
                old_weight = self.weighted_map[start_vertex, target_vertex]
                self.dist_matrix[start_vertex, target_vertex] = 0
                self.traffic_matrix[start_vertex, target_vertex] = 0
                self.weighted_map[start_vertex, target_vertex] = 0
                self.dist_matrix[target_vertex, start_vertex] = 0
                self.traffic_matrix[target_vertex, start_vertex] = 0
                self.weighted_map[target_vertex, start_vertex] = 0
                no_path = not (self.validate_accessibility(start_vertex, target_vertex) and
                               self.validate_accessibility(target_vertex, start_vertex))
                if no_path:
                    self.dist_matrix[start_vertex, target_vertex] = old_dist
                    self.traffic_matrix[start_vertex, target_vertex] = old_traffic
                    self.weighted_map[start_vertex, target_vertex] = old_weight
                    self.dist_matrix[target_vertex, start_vertex] = old_dist
                    self.traffic_matrix[target_vertex, start_vertex] = old_traffic
                    self.weighted_map[target_vertex, start_vertex] = old_weight
                counter += 1

            used_points += [start_vertex, target_vertex]
        logging.debug(f'Construction sites:\n {used_points}, amount: {self.num_construction_sites}')

    def get_map(self):
        nodes = self.height * self.width
        map_matrix = np.zeros((nodes, nodes))
        for i in range(nodes):
            for j in range(nodes):
                map_matrix[i, j] = round(self.dist_matrix[i, j] * self.traffic_matrix[i, j], 2)
        return map_matrix

    def validate_accessibility(self, start_vertex, target_vertex):
        if start_vertex == target_vertex:
            return True
        queue = [start_vertex]
        explored = []
        while len(queue):
            vertex = queue.pop()
            explored.append(vertex)
            for next_vertex in np.argwhere(self.dist_matrix[:, vertex] > 0).reshape(-1):
                if next_vertex in explored:
                    continue
                if next_vertex == target_vertex:
                    return True
                explored.append(next_vertex)
                queue.append(next_vertex)
        return False

    def draw_map(self):
        g = nx.DiGraph()
        pos = {}
        for v, p in zip(self.vertices_matrix.reshape(-1).tolist(), [
            (x, y) for y in range(self.height - 1, -1, -1) for x in range(self.width)
        ]):
            pos[v] = p
            for next_v in np.argwhere(self.dist_matrix[:, v] > 0).reshape(-1):
                weight = self.weighted_map[next_v, v]
                g.add_edge(v, next_v, weight=weight)
        traffic_lights = [self.vertices_matrix[light] for light in self.traffic_lights]
        packages = [self.vertices_matrix[package] for package in self.packages]
        nx.draw_networkx(g, pos)
        nx.draw_networkx_nodes(g, pos, nodelist=traffic_lights, node_color="yellow")
        nx.draw_networkx_nodes(g, pos, nodelist=packages, node_color="red")
        nx.draw_networkx_nodes(g, pos, nodelist=[self.vertices_matrix[self.pos]], node_color="green")
        labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, labels)
        plt.show()
