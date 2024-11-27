import numpy as np
from nasbench import api as ModelSpecAPI

from monet.search_spaces.nasbench201_node import NASBench201Cell
from naslib.search_spaces.core import Metric
from naslib.utils import get_dataset_api


class NASBench101Vertice:

    def __init__(self, id):
        self.id = id
        self.label = "none"
        self.edges = {i: 0 for i in range(id)}  # 0 ou 1 : connexion avec les autres vertices
        self.OPERATIONS = ["none", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3", "input", "output"]
        self.playable_operations = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]  # Les labels qu'on peut assigner

    def get_action_tuples(self):
        list_tuples = []
        if self.label == "none":
            for op in self.playable_operations:
                list_tuples.append(("set_label", op))
        for k, v in self.edges.items():
            if v == 0:
                list_tuples.append(("build_edge", k))
        return list_tuples

    def play_action(self, action_name, action):
        if action_name == "set_label":
            self.label = action
        elif action_name == "build_edge":
            k = action
            self.edges[k] = 1


class NASBench101Cell(NASBench201Cell):

    def __init__(self, n_vertices, vertice_type=NASBench101Vertice):
        super().__init__(n_vertices, vertice_type)

        self.vertices[0].play_action("set_label", "input")
        self.vertices[1].play_action("build_edge", 0)
        self.vertices[n_vertices - 1].play_action("set_label", "output")
        self.vertices[n_vertices - 1].play_action("build_edge", n_vertices-2)

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, connexion in vertice.edges.items():
                if connexion == 1:
                    adjacency_matrix[j, i] = 1
        return adjacency_matrix

    def operations_and_adjacency(self):
        adjacency = self.adjacency_matrix()
        operations = []
        for v in self.vertices:
            operations.append(v.label)

        return adjacency, operations

    def play_action(self, vertice, id, operation):

        self.vertices[vertice].play_action(id, operation)
        super().play_action(vertice, id,  operation)

    def get_action_tuples(self):
        sum_edges = 0
        for v in self.vertices:
            n_edges = int(np.sum(list(v.edges.values())))
            sum_edges += n_edges
        list_tuples = []
        for i, v in enumerate(self.vertices):
            actions = v.get_action_tuples()
            if sum_edges >= 9:
                actions_dup = []
                for act in actions:
                    if act[0] == "set_label":
                        actions_dup.append(act)
                actions = actions_dup
            for action in actions:
                list_tuples.append((i, *action))
        return list_tuples


    def is_complete(self):
        is_complete = True
        sum_edges = 0
        for v in self.vertices:
            n_edges = int(np.sum(list(v.edges.values())))
            sum_edges += n_edges
        if sum_edges > 9:
            is_complete = False
        for v in self.vertices:
            if v.label == "none":
                is_complete = False
        return is_complete

    def calculate_zobrist_hash(self, zobrist_table):
        hash = 0
        adjacency = self.adjacency_matrix()
        for i, element in enumerate(adjacency.flatten()):
            hash ^= zobrist_table[i][element]
        for i, v in enumerate(self.vertices):
            op_index = v.OPERATIONS.index(v.label)
            hash ^= zobrist_table[adjacency.shape[0]**2+i][op_index]
        return hash

    def get_reward(self, api, metric=Metric.VAL_ACCURACY, dataset="cifar10", df=None):
        assert metric.name in ["VAL_ACCURACY"], f"Only VAL_ACCURACY is supported, not {metric.name}"
        adjacency, operations = self.operations_and_adjacency()
        model_spec = ModelSpecAPI.ModelSpec(
                # Adjacency matrix of the module
                matrix=adjacency,
                # Operations at the vertices of the module, matches order of matrix
                ops=operations)
        if not model_spec.valid_spec:
            # INVALID SPEC
            return 0
        if metric.name == "VAL_ACCURACY":
            reward = api.query(model_spec)["validation_accuracy"] * 100
        return reward
