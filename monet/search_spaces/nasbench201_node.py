import random
from copy import deepcopy

import numpy as np
from graphviz import Digraph

from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices, convert_op_indices_to_naslib
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric


class NASBench201Vertice:
    """
    Class representing a single vertice in the NASBench201 search space.
    """

    def __init__(self, id):
        """
        Initialize a NASBench201Vertice instance.

        Parameters:
        id (int): The ID of the vertice.
        """
        self.id = id
        self.actions = {i: None for i in range(id)}
        self.OPERATIONS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

    def is_complete(self):
        """
        Check if all actions for the vertice are defined.

        Returns:
        bool: True if all actions are defined, False otherwise.
        """
        return None not in self.actions.values()

    def play_action(self, id, operation):
        """
        Define an action for the vertice.

        Parameters:
        id (int): The ID of the input vertice.
        operation (str): The operation to be performed.
        """
        self.actions[id] = operation

    def get_action_tuples(self):
        """
        Get all possible action tuples for the vertice.

        Returns:
        list: A list of tuples representing possible actions.
        """
        list_tuples = []
        for k, v in self.actions.items():
            if v is None:
                for op in self.OPERATIONS:
                    list_tuples.append((k, op))
        return list_tuples


class NASBench201Cell:
    """
    Class representing a cell in the NASBench201 search space.
    """

    def __init__(self, n_vertices=4, vertice_type=NASBench201Vertice):
        """
        Initialize a NASBench201Cell instance.

        Parameters:
        n_vertices (int): The number of vertices in the cell.
        vertice_type (class): The class type for vertices.
        """
        self.n_vertices = n_vertices
        self.vertices = [vertice_type(i) for i in range(n_vertices)]
        self.OPERATIONS = self.vertices[0].OPERATIONS
        self.zobrist_table = None
        self.N_NODES = 4
        self.ADJACENCY_MATRIX_SIZE = self.N_NODES ** 2
        self.N_OPERATIONS = 5

    def is_complete(self):
        """
        Check if all vertices in the cell are complete.

        Returns:
        bool: True if all vertices are complete, False otherwise.
        """
        return all([v.is_complete() for v in self.vertices])

    def to_str(self):
        """
        Convert the cell to a string representation.

        Returns:
        str: The string representation of the cell.
        """
        assert self.is_complete(), "cell is incomplete"
        res = ""
        for v in self.vertices[1:]:
            temp_str = "|"
            # sorted_actions = collections.OrderedDict(sorted(d.items()))
            for source, operation in v.actions.items():
                temp_str += f"{operation}~{source}|"
            res += temp_str + "+"
        return res[:-1]  # -1 pour enlever le dernier "+"

    def adjacency_matrix(self):
        """
        Generate the adjacency matrix for the cell.

        Returns:
        np.ndarray: The adjacency matrix of the cell.
        """
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, operation in vertice.actions.items():
                if operation is None:
                    operation = "none"
                adjacency_matrix[j,i] = self.OPERATIONS.index(operation)
        return adjacency_matrix

    def play_action(self, vertice, id, operation):
        """
        Define an action for a specific vertice in the cell.

        Parameters:
        vertice (int): The vertice that plays the action.
        id (int): The vertice that acts as input for the action.
        operation (str): The operation to be performed.
        """
        self.vertices[vertice].play_action(id, operation)

    def get_action_tuples(self):
        """
        Get all possible action tuples for the cell.

        Returns:
        list: A list of tuples representing possible actions.
        """
        list_tuples = []
        for i, v in enumerate(self.vertices):
            actions = v.get_action_tuples()
            for action in actions:
                list_tuples.append((i, *action))
        return list_tuples

    def calculate_zobrist_hash(self, zobrist_table):
        """
        Calculate the Zobrist hash for the cell.

        Parameters:
        zobrist_table (list): The Zobrist table used for hashing.

        Returns:
        int: The Zobrist hash of the cell.
        """
        assert zobrist_table is not None, "Remember to pass zobrist_table to node constructor."
        hash = 0
        adjacency = self.adjacency_matrix()
        for i, row in enumerate(adjacency):
            for element in row:
                hash ^= zobrist_table[i][element]
        return hash

    def initialize_zobrist_table(self):
        """
        Initialize the Zobrist table for the cell.
        """
        self.zobrist_table = []
        for i in range(self.ADJACENCY_MATRIX_SIZE):
            adjacency_table = []
            for connexion in range(2):  # Connextion or no connexion
                adjacency_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(adjacency_table)
        for i in range(self.N_NODES):
            operation_table = []
            for operation in range(self.N_OPERATIONS):
                operation_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(operation_table)

    def get_reward(self, api, metric=Metric.VAL_ACCURACY, dataset="cifar10", df=None):
        """
        Fetch the reward for the cell using NASLib.

        Parameters:
        api: The NASLib API instance.
        metric (Metric): The metric to be used for evaluation.
        dataset (str): The dataset to be used for evaluation.
        df (pd.DataFrame): The DataFrame containing architecture scores.

        Returns:
        float: The reward for the cell.
        """

        if df is not None:
            assert metric == Metric.VAL_ACCURACY, "Only VAL_ACCURACY is supported for now."
            assert dataset == "cifar10", "Only CIFAR-10 is supported for now."
            if metric == Metric.VAL_ACCURACY and dataset == "cifar10":
                metric_to_fetch = "cifar_10_val_accuracy"
            arch_str = self.to_str()
            row = df.loc[df["arch_str"] == arch_str]
            reward = row[metric_to_fetch].item()
            return reward

        arch_str = self.to_str()
        indices = convert_str_to_op_indices(arch_str)
        candidate = NasBench201SearchSpace()
        convert_op_indices_to_naslib(indices, candidate)
        reward = candidate.query(dataset=dataset, metric=metric, dataset_api=api)

        return reward


    def plot(self, filename="cell"):
        """
        Plot the cell using Graphviz.

        Parameters:
        filename (str): The filename for the output plot.
        """
        g = Digraph(
                format='pdf',
                edge_attr=dict(fontsize='20', fontname="garamond"),
                node_attr=dict(style='rounded, filled', shape='rect', align='center', fontsize='20', height='0.5',
                               width='0.5', penwidth='2', fontname="garamond"),
                engine='dot')
        g.body.extend(['rankdir=LR'])

        g.node("c_{k-1}", fillcolor='darkseagreen2')
        g.node("c_{k}", fillcolor='palegoldenrod')

        steps = self.n_vertices - 2

        for i in range(steps):
            g.node(str(i + 1), fillcolor='lightblue')

        for i, vertice in enumerate(self.vertices):
            for k, v in vertice.actions.items():
                # print(str(i), str(k), v)
                in_ = str(k)
                out_ = str(i)
                if k == 0:
                    in_ = "c_{k-1}"
                if i == self.n_vertices - 1:
                    out_ = "c_{k}"
                g.edge(in_, out_, label=v, fillcolor="gray")

        g.render(filename, view=True)

if __name__ == '__main__':
    from naslib.utils import get_dataset_api
    from naslib.search_spaces.core.query_metrics import Metric
    dataset_api = get_dataset_api("nasbench201", 'cifar10')

    cell = NASBench201Cell()
    cell.initialize_zobrist_table()
    print(cell.calculate_zobrist_hash(cell.zobrist_table))
    # Complete cell
    while not cell.is_complete():
        actions = cell.get_action_tuples()
        action = random.choice(actions)
        cell.play_action(*action)
    reward = cell.get_reward(dataset_api)
    print(reward)