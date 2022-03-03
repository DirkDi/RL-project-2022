import numpy as np
import gym

import logging
import random
import networkx as nx
import matplotlib.pyplot as plt

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class CityEnv(gym.Env):
    """
    An environment to simulate a city traffic.
    """

    def __init__(self, height=3, width=3, min_distance=10, max_distance=100, min_traffic=1, max_traffic=2,
                 dist_matrix: np.ndarray = None,
                 traffic_matrix: np.ndarray = None,
                 packages=None, num_packages: int = 2,
                 init_random=False, one_way=True, construction_sites=True,
                 traffic_lights=True):
        """
        Initializes the environment.
        """
        super(CityEnv, self).__init__()
        # throw error message if environment is not possible
        assert np.all(np.array([height, width, min_distance, max_distance,
                                min_traffic, max_traffic, num_packages]) > 0), "all arguments must be non-negative!"
        assert min_distance < max_distance and min_traffic < max_traffic, \
            "minimum values have to be lower than maximum values!"
        assert (dist_matrix is not None and traffic_matrix is not None) or \
               (dist_matrix is None and traffic_matrix is None), "if one matrix is defined the other must be too!"
        if dist_matrix is not None:
            assert np.all(dist_matrix >= 0), "all entries in the distance matrix must not be non-negative!"
        if traffic_matrix is not None:
            assert np.all(traffic_matrix >= 0), "all entries in the traffic matrix must not be non-negative!"
        if dist_matrix is not None and traffic_matrix is not None:
            # check if distance and traffic matrix have the same dimension
            assert dist_matrix.shape == traffic_matrix.shape, \
                f"traffic and distance matrix need to have the same dimension!"
            # check if distance and traffic matrix have the same edges
            dist_idx = np.argwhere(dist_matrix > 0)
            traffic_idx = np.argwhere(traffic_matrix > 0)
            assert np.array_equal(dist_idx, traffic_idx), "traffic and distance matrix need to have the same edges!"
            matrix_height = height * width
            assert (matrix_height == dist_matrix.shape[0]) and (matrix_height == dist_matrix.shape[1]) and \
                   (matrix_height == traffic_matrix.shape[0]) and (matrix_height == traffic_matrix.shape[1]), \
                   "given height and width do not match distance matrix length!"

        self.height = height
        self.width = width

        self.min_distance = min_distance  # minimum distance between vertices
        self.max_distance = max_distance  # maximum distance between vertices
        self.min_traffic = min_traffic  # minimum traffic occurrence between vertices
        self.max_traffic = max_traffic  # maximum traffic occurrence between vertices

        # Create vertices matrix
        self.matrix_height = self.height * self.width
        self.vertices_matrix = np.reshape(np.arange(0, self.matrix_height), (-1, self.width))

        # create distance matrix if it is None
        if dist_matrix is None:
            dist_matrix = np.zeros((self.matrix_height, self.matrix_height))
            for i in range(height):
                for j in range(width):
                    start_vertex = self.vertices_matrix[i, j]
                    for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= a < height and 0 <= b < width:
                            target_vertex = self.vertices_matrix[a, b]
                            if dist_matrix[start_vertex, target_vertex] > 0:
                                continue
                            dist = random.randint(min_distance, max_distance) if init_random else 1
                            dist_matrix[start_vertex, target_vertex] = dist
                            dist_matrix[target_vertex, start_vertex] = dist
        logging.info("dist_matrix created")

        # create traffic matrix if it is None
        if traffic_matrix is None:
            traffic_matrix = np.zeros((self.matrix_height, self.matrix_height))
            for i in range(height):
                for j in range(width):
                    start_vertex = self.vertices_matrix[i, j]
                    for a, b in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if 0 <= a < height and 0 <= b < width:
                            target_vertex = self.vertices_matrix[a, b]
                            if traffic_matrix[start_vertex, target_vertex] > 0:
                                continue
                            flow = round(random.uniform(min_traffic, max_traffic), 2) if init_random else 1
                            traffic_matrix[start_vertex, target_vertex] = flow
                            traffic_matrix[target_vertex, start_vertex] = flow
        logging.info("traffic_matrix created")

        self.dist_matrix = dist_matrix.copy()
        self.traffic_matrix = traffic_matrix.copy()

        # Check if city graph is connected
        for i in range(self.matrix_height):
            for j in range(self.matrix_height):
                assert self.validate_accessibility(i, j), f"{self.matrix_height} The city graph is not for nodes {i} and {j} connected!"

        minimal_generating = min(self.height - 1, self.width - 1)
        if minimal_generating:
            self.num_one_way = random.randint(1, min(self.height - 1, self.width - 1))
            self.num_construction_sites = random.randint(1, min(self.height - 1, self.width - 1))
            self.num_traffic_lights = random.randint(1, self.width * self.height // 2)
        else:
            self.num_one_way = 0
            self.num_construction_sites = 0
            self.num_traffic_lights = 0
        self.traffic_lights = []  # list of coordinates with traffic lights

        if packages is None:
            assert init_random, "If no packages are defined, init_random must be set to True!"
            packages = []
            self.num_packages = num_packages
            for i in range(num_packages):
                packages.append((random.randint(0, self.height - 1), random.randint(0, self.width - 1)))
        else:
            self.num_packages = len(packages)
        self.packages = packages.copy()
        self.packages_initial = packages.copy()
        logging.debug(f'Coordinates of packages are: {packages}')

        while True:
            self.init_pos = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            if self.init_pos not in self.packages:
                self.pos = self.init_pos
                self.prev_pos = self.init_pos
                break
        logging.debug(f'The start position is {self.init_pos}')

        self.weighted_map = self.get_map()
        logging.info("weighted_map created")
        if one_way:
            self.generate_one_way_streets()
        logging.info("one way streets created")
        if construction_sites:
            self.generate_construction_sites()
        logging.info("construction sites created")
        if traffic_lights:
            self.generate_traffic_lights()
        logging.debug(f'Distance matrix:\n{dist_matrix}')
        logging.debug(f'Traffic matrix:\n{traffic_matrix}')
        logging.debug(f'Weighted map matrix:\n{self.weighted_map}')

        low = np.array([0, 0, 0])
        high = np.array([self.height, self.width, num_packages])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.reward_range = [float("-inf"), 0]  # steps are negative rewards so the best possible reward is 0
        self.already_driven = [self.pos]  # contains the points where the agent already was

    def reset(self):
        """
        Resets the environment to the initial state
        """
        self.pos = self.init_pos
        self.packages = self.packages_initial.copy()
        return np.array([self.pos[0], self.pos[1], len(self.packages)]).astype(dtype=np.int32)

    def step(self, action):
        """
        Performs a step on the environment and calculates the specific reward of this step.
        """
        action = int(action)
        if action < 0 or action >= 4:
            raise RuntimeError(f"{action} is not a valid action (needs to be between 0 and 3)")

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
            reward = -1000  # -10000 * (self.height * self.width)
            return np.array([pos_x, pos_y, len(self.packages)]).astype(np.int32), reward, False, {
                'render.modes': ['console']}
        start_vertex = self.vertices_matrix[pos_x, pos_y]
        target_vertex = self.vertices_matrix[new_pos_x, new_pos_y]
        dist = self.dist_matrix[target_vertex, start_vertex]
        traffic_flow = self.traffic_matrix[target_vertex, start_vertex]
        # action is not allowed if there is no vertex between both points (value is 0 for dist & traffic_flow)
        if not dist:
            reward = -1000  # -10000 * (self.height * self.width)
            return np.array([pos_x, pos_y, len(self.packages)]).astype(np.int32), reward, False, {
                'render.modes': ['console']}
        self.pos = new_pos_x, new_pos_y
        reward = -(dist * traffic_flow) if self.pos not in self.traffic_lights else -(
                dist * traffic_flow) * 1.2
        if (new_pos_x, new_pos_y) in self.packages:
            while (new_pos_x, new_pos_y) in self.packages:
                self.packages.remove((new_pos_x, new_pos_y))
            self.already_driven = []  # reset already driven array

        packages_count = len(self.packages)
        done = packages_count == 0

        meta_info = {'render.modes': ['console']}
        self.already_driven.append((new_pos_x, new_pos_y))
        # logging.debug(self.already_driven)
        return np.array([new_pos_x, new_pos_y, packages_count]).astype(np.int32), reward, done, meta_info

    def close(self):
        """
        Makes sure that the environment is closed.
        """
        pass

    def render(self, mode="human"):
        """
        Renders the environment.
        """
        pass

    def generate_traffic_lights(self):
        """
        Randomly choose positions where traffic lights are which make the node less efficient.
        """
        for i in range(self.num_traffic_lights):
            self.traffic_lights.append((random.randint(0, self.height - 1), random.randint(0, self.width - 1)))
        logging.debug(f'Traffic lights:\n {self.traffic_lights}, amount: {self.num_traffic_lights}')

    def generate_one_way_streets(self):
        """
        Generates one-way streets by randomly transforming bidirectional edges to unidirectional ones,
        always maintaining reachability to all nodes.
        """
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
        """
        Generates construction sites by randomly removing edges,
        always maintaining reachability to all nodes.
        """
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
        """
        Computes the weight matrix by multiplying the distance matrix with the traffic matrix
        """
        return np.round(self.dist_matrix * self.traffic_matrix, 2)

    def validate_accessibility(self, start_vertex, target_vertex):
        """
        Help-function to realise one-way streets and construction sites.
        This function is used to check the accessibility/reachability from one node
        to another ( a path will be searched).

        :param start_vertex: Start point to find a path for.
        :param target_vertex: End point to find a path for.

        :return: boolean where True represents that a path was found and False represents that no path was found.
        """
        # negative indices are not allowed (no path exists)
        if start_vertex < 0 or target_vertex < 0:
            return False
        # nodes out of range are not allowed (no path exists)
        max_node = self.vertices_matrix.shape[0] * self.vertices_matrix.shape[1] - 1
        if start_vertex > max_node or target_vertex > max_node:
            return False
        # always true because start is also the end
        if start_vertex == target_vertex:
            return True
        queue = [start_vertex]
        explored = []
        while queue:
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
        """
        Shows a graphic representation of the city graph.

        Note that you have to close the window manually
        before the program will go on.
        """
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

    def get_min_emission_action(self):
        """
        Chooses the action with the lowest edge weight. If all possible edges from a node are already driven the function
        chooses randomly a path to avoid an endless loop.
        """
        i, j = self.pos
        start_vertex = self.vertices_matrix[i, j]
        action, min_weight = -1, float('inf')
        actions = []
        for i, (a, b) in enumerate([(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]):
            # check for valid coordinates
            if 0 <= a < self.height and 0 <= b < self.width:
                target_vertex = self.vertices_matrix[a, b]
                weight = self.weighted_map[target_vertex, start_vertex]
                if weight > 0:
                    actions.append(i)
                    if (a, b) in self.traffic_lights:
                        weight *= 1.2
                    # Update the weight if a better weight was found.
                    # It also should not be already visited to avoid endless loops
                    if weight < min_weight and (a, b) not in self.already_driven:
                        min_weight = weight
                        action = i
        # If no action was chosen, choose a random one which however is as valid
        # as the agent moves forward and doesn't stay on the same place.
        return action if action != -1 else np.random.choice(actions)

    def get_max_emission_action(self):
        """
        Chooses the action with the highest edge weight. If all possible edges from a node are already driven
        the function chooses randomly a path to avoid an endless loop.
        """
        i, j = self.pos
        start_vertex = self.vertices_matrix[i, j]
        action, min_weight = -1, float('-inf')
        actions = []
        for i, (a, b) in enumerate([(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]):
            # check for valid coordinates
            if 0 <= a < self.height and 0 <= b < self.width:
                target_vertex = self.vertices_matrix[a, b]
                weight = self.weighted_map[target_vertex, start_vertex]
                if weight > 0:
                    actions.append(i)
                    if (a, b) in self.traffic_lights:
                        weight *= 1.2
                    # Update the weight if a better weight was found.
                    # It also should not be already visited to avoid endless loops
                    if weight > min_weight and (a, b) not in self.already_driven:
                        min_weight = weight
                        action = i
        # If no action was chosen, choose a random one which however is as valid
        # as the agent moves forward and doesn't stay on the same place.
        return action if action != -1 else np.random.choice(actions)
