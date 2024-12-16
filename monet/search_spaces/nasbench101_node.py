import copy
import random

import numpy as np
from nasbench import api as ModelSpecAPI

from monet.search_spaces.nasbench201_node import NASBench201Cell
from naslib.search_spaces.core import Metric
from naslib.search_spaces.nasbench101.conversions import convert_spec_to_tuple
from naslib.utils import get_dataset_api
from naslib.utils.nb101_api import hash_module


class NASBench101Vertice:

    def __init__(self, id):
        self.id = id
        self.label = "none"
        self.edges = {i: 0 for i in range(id)}  # 0 ou 1 : connexion avec les autres vertices
        self.OPERATIONS = ["none", 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', "input", "output"]
        self.playable_operations = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']  # Les labels qu'on peut assigner

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
        self.N_NODES = 7
        self.ADJACENCY_MATRIX_SIZE = self.N_NODES ** 2
        self.N_OPERATIONS = 6
        self.playable_operations = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']  # Les labels qu'on peut assigner


    def get_hash(self):
        return convert_spec_to_tuple({"matrix": self.adjacency_matrix(), "ops": [v.label for v in self.vertices]})

    def hash_cell(self):
        pruned_matrix, pruned_operations = self.prune()  # Getting the reduced rank matrix and operations
        # Getting the labeling used in naslib/utils/nb101_api
        labeling = [-1] + [self.vertices[0].playable_operations.index(op) for op in pruned_operations[1:-1]] + [-2]
        hash = hash_module(pruned_matrix, labeling)
        return hash

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

        if df is not None:
            assert metric == Metric.VAL_ACCURACY, "Only VAL_ACCURACY is supported for now."
            assert dataset == "cifar10", "Only CIFAR-10 is supported for now."
            if metric == Metric.VAL_ACCURACY and dataset == "cifar10":
                metric_to_fetch = "cifar_10_val_accuracy"
            if self.prune() is None:  # INVALID SPEC
                return 0
            arch_hash = self.hash_cell()
            row = df.loc[df["arch_hash"] == arch_hash]
            reward = row[metric_to_fetch].item()
            return reward

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

    def prune(self):
        """Prune the extraneous parts of the graph.

        General procedure:
          1) Remove parts of graph not connected to input.
          2) Remove parts of graph not connected to output.
          3) Reorder the vertices so that they are consecutive after steps 1 and 2.

        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        adjacency_matrix, ops = self.operations_and_adjacency()
        num_vertices = np.shape(adjacency_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if adjacency_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if adjacency_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            matrix = None
            ops = None
            valid_spec = False
            return

        adjacency_matrix = np.delete(adjacency_matrix, list(extraneous), axis=0)
        adjacency_matrix = np.delete(adjacency_matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del ops[index]
        return adjacency_matrix, ops

    def mutate(self, api, mutation_rate=1):
        original_matrix, original_ops = copy.deepcopy(self.operations_and_adjacency())
        while True:
            new_matrix, new_ops = copy.deepcopy(self.operations_and_adjacency())
            edge_mutation_probability = mutation_rate / self.N_NODES
            for src in range(0, self.N_NODES-1):
                for dst in range(src+1, self.N_NODES):
                    if random.random() < edge_mutation_probability:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            operation_mutation_probability = mutation_rate / self.N_OPERATIONS
            for ind in range(0, self.N_NODES-1):
                if random.random() < operation_mutation_probability:
                    available = [op for op in self.playable_operations if op != new_ops[ind]]
                    new_ops[ind] = np.random.choice(self.vertices[0].playable_operations)
            new_spec = ModelSpecAPI.ModelSpec(
                # Adjacency matrix of the module
                matrix=new_matrix,
                # Operations at the vertices of the module, matches order of matrix
                ops=new_ops)
            if api.is_valid(new_spec):
                break
        for i, vertice in enumerate(self.vertices):
            for j, connexion in vertice.edges.items():
                vertice.edges[j] = new_matrix[j, i]
            vertice.label = new_ops[i]