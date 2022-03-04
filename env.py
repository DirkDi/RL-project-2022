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
    An environment to simulate city traffic.
    """

    def __init__(self, height=3, width=3, min_distance=10, max_distance=100, min_traffic=1, max_traffic=2,
                 dist_matrix: np.ndarray = None,
                 traffic_matrix: np.ndarray = None,
                 traffic_lights_list: list = None,
                 init_pos: tuple = None, packages=None, num_packages: int = 2,
                 init_random=False, one_way=True, construction_sites=True,
                 traffic_lights=True):
        """
        Initializes the environment.

        :param height: the height of the environment (height of weighted map)
        :param width: the width of the environment (width of weighted map)
        :param min_distance: the minimum length a street can have (distance between nodes/crossings)
        :param max_distance: the maximum length a street can have (distance between nodes/crossings)
        :param min_traffic: the minimum traffic flow factor a street can have
        :param max_traffic: the maximum traffic flow factor a street can have
        :param dist_matrix: the distance matrix which contains the length between all connected crossings
        :param traffic_matrix: the traffic matrix which contains the traffic flow between all connected crossings
        :param traffic_lights_list: The list of manually created traffic lights
        :param init_pos: the manually chosen initial position of the car
        :param packages: the manually chosen packages (list of coordinates) that need to be delivered
        :param num_packages: the amount of packages which should be generated randomly
        :param init_random: activates/deactivates random numbers for distance and traffic matrix (non-random is 1)
        :param one_way: activates/deactivates the random creation of one-way streets
        :param construction_sites: activates/deactivates the random creation of construction sites
        :param traffic_lights: activates/deactivates the random creation of traffic lights
        """
        super(CityEnv, self).__init__()
        # throw error message if environment is not possible
        assert np.all(np.array([height, width, min_distance, max_distance,
                                min_traffic, max_traffic, num_packages]) > 0), "all arguments must be non-negative!"
        assert min_distance < max_distance and min_traffic < max_traffic, \
            "minimum values have to be lower than maximum values!"
        assert (dist_matrix is not None and traffic_matrix is not None) or \
               (not dist_matrix and not traffic_matrix), "if one matrix is defined the other must be too!"
        if dist_matrix is not None:
            assert np.all(dist_matrix >= 0), "all entries in the distance matrix must be non-negative!"
        if traffic_matrix is not None:
            assert np.all(traffic_matrix >= 0), "all entries in the traffic matrix must be non-negative!"
        if dist_matrix is not None and traffic_matrix is not None:
            # check if distance and traffic matrix have the same dimension
            assert dist_matrix.shape == traffic_matrix.shape, \
                f"traffic and distance matrix need to have the same dimension!"
            # check if distance and traffic matrix have the same edges
            dist_idx = np.argwhere(dist_matrix > 0)
            traffic_idx = np.argwhere(traffic_matrix > 0)
            assert np.array_equal(dist_idx, traffic_idx), "traffic and distance matrices need to have the same edges!"
            matrix_height = height * width
            assert (matrix_height == dist_matrix.shape[0]) and (matrix_height == dist_matrix.shape[1]) and \
                   (matrix_height == traffic_matrix.shape[0]) and (matrix_height == traffic_matrix.shape[1]), \
                "given height and width do not match distance matrix length!"

        self.CO2 = 0.142  # constant for co2 emission per meter (diesel car)
        self.PENALTY = -250  # reward penalty for illegal action
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
                assert self.validate_accessibility(i, j), f"Can not reach node {j} from node {i}."

        # generates constraints based on min of height / width as  possible maximum number per constraint
        minimal_generating = min(self.height - 1, self.width - 1)
        if minimal_generating:
            self.num_one_way = random.randint(1, min(self.height - 1, self.width - 1))
            self.num_construction_sites = random.randint(1, min(self.height - 1, self.width - 1))
            self.num_traffic_lights = random.randint(1, self.width * self.height // 2)
        else:
            self.num_one_way = 0
            self.num_construction_sites = 0
            self.num_traffic_lights = 0

        # check if manually given traffic lights are correctly placed
        if traffic_lights_list:
            correct_placed = True
            for traffic_light in traffic_lights_list:
                if traffic_light[0] >= self.height or traffic_light[1] >= self.width:
                    correct_placed = False
                    break
            assert correct_placed, "traffic lights are out of bound!"

        # list of coordinates with traffic lights
        self.traffic_lights = traffic_lights_list if traffic_lights_list else []

        # randomly generate packages if there are no manually given packages
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

        # check if initial pos is correctly placed if manually given
        if init_pos:
            assert init_pos[0] < self.height and init_pos[1] < self.width, "initial position is out of bound!"
            if packages:
                assert init_pos not in packages, "initial position has to be placed on an empty node (no package)!"

        # generate initialization position randomly but not the same place like a package
        else:
            while True:
                init_pos = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
                if init_pos not in self.packages:
                    break

        self.init_pos = init_pos
        self.pos = self.init_pos
        logging.debug(f'The starting position is {self.init_pos}')

        self.weighted_map = self.get_map()
        logging.info("weighted_map created")
        if one_way:
            self.generate_one_way_streets()
        logging.info("one way streets created")
        if construction_sites:
            self.generate_construction_sites()
        logging.info("construction sites created")
        if traffic_lights and not traffic_lights_list:
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

        :return: returns the observation space
        """
        self.pos = self.init_pos
        self.packages = self.packages_initial.copy()
        return np.array([self.pos[0], self.pos[1], len(self.packages)]).astype(dtype=np.int32)

    def step(self, action):
        """
        Performs a step on the environment and calculates the specific reward of this step.

        :param action: the action to take for the current step

        :return: observation space
        :return: reward for the taken action
        :return: done flag, true when all packages have been delivered
        :return: meta info, contains additional information like render mode
        """
        action = int(action)
        if action < 0 or action >= 4:
            raise RuntimeError(f"{action} is not a valid action (needs to be between 0 and 3)")

        # change car position based on action
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
            reward = self.PENALTY
            return np.array([pos_x, pos_y, len(self.packages)]).astype(np.int32), reward, False, {
                'render.modes': ['console']}
        start_vertex = self.vertices_matrix[pos_x, pos_y]
        target_vertex = self.vertices_matrix[new_pos_x, new_pos_y]
        dist = self.dist_matrix[target_vertex, start_vertex]
        traffic_flow = self.traffic_matrix[target_vertex, start_vertex]

        # action is not allowed if there is no edge between both points (value is 0 for dist & traffic_flow)
        if not dist:
            reward = self.PENALTY
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
        if (new_pos_x, new_pos_y) in self.already_driven:
            reward = self.PENALTY
            return np.array([new_pos_x, new_pos_y, packages_count]).astype(np.int32), reward, done, {
                'render.modes': ['console']}
        self.already_driven.append((new_pos_x, new_pos_y))
        return np.array([new_pos_x, new_pos_y, packages_count]).astype(np.int32), reward * self.CO2, done, meta_info

    def close(self):
        """
        Makes sure that the environment is closed.
        """
        pass

    def render(self, mode="human"):
        """
        Renders the environment.
        :param mode: the mode for rendering
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

        :return: calculated weighted map of city
        """
        return np.round(self.dist_matrix * self.traffic_matrix, 2)

    def validate_accessibility(self, start_vertex, target_vertex):
        """
        Help-function to realise one-way streets and construction sites.
        This function is used to check the accessibility/reachability from one node
        to another.

        :param start_vertex: Start point to find a path from.
        :param target_vertex: End point to find a path to.

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
            (x, y) for y in range(self.height - 1, -1, -1) for x in range(self.width)]
        ):
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
        Chooses the action with the lowest edge weight. If all possible edges from a node are already driven the
        function chooses a path randomly to avoid an endless loop.

        :return: the next action for the agent inside the environment
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
                    weight *= self.CO2
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
        the function chooses a path randomly to avoid an endless loop.

        :return: the next action for the agent inside the environment
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
                    weight *= self.CO2
                    # Update the weight if a better weight was found.
                    # It also should not be already visited to avoid endless loops
                    if weight > min_weight and (a, b) not in self.already_driven:
                        min_weight = weight
                        action = i
        # If no action was chosen, choose a random one which however is as valid
        # as the agent moves forward and doesn't stay on the same place.
        return action if action != -1 else np.random.choice(actions)
