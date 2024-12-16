import copy
import itertools
import os
from collections import namedtuple
import random
from typing import Tuple

import nasbench301 as nb  # NB301
import numpy as np

from graphviz import Digraph
from naslib.search_spaces.nasbench301.conversions import convert_genotype_to_naslib, convert_genotype_to_compact, \
    make_compact_immutable, convert_compact_to_genotype, make_compact_mutable

from naslib.search_spaces import NasBench301SearchSpace

from naslib.search_spaces.core import Metric

from monet.search_spaces.nasbench201_node import NASBench201Cell, NASBench201Vertice

N_NODES = 6
ADJACENCY_MATRIX_SIZE = N_NODES ** 2
N_OPERATIONS = 7

class DARTSVertice(NASBench201Vertice):

    def __init__(self, id):
        super().__init__(id)
        self.id = id
        if id == 1:  # Il y a deux input cells
            self.actions = {}
        else:
            self.actions = {i: "none" for i in
                            range(id)}  # la valeur est l'opération qu'on joue entre deux représentations
        self.OPERATIONS = ["max_pool_3x3", "avg_pool_3x3", "skip_connect", "sep_conv_3x3",
                           "sep_conv_5x5", "dil_conv_3x3", "dil_conv_5x5"]  # Les labels qu'on peut assigner

    def get_n_predecessors(self):
        actions = [i for i in self.actions.values() if i != "none"]
        return len(actions)

    def get_action_tuples(self):
        n_predecessors = self.get_n_predecessors()
        if self.id > 1 and n_predecessors < 2:
            list_tuples = []
            for k, v in self.actions.items():
                if v == "none":
                    for op in self.OPERATIONS:
                        list_tuples.append((k, op))
            return list_tuples


    def is_complete(self):
        if self.id > 1:
            n_predecessors = self.get_n_predecessors()
            is_complete = (n_predecessors == 2)
            return is_complete
        else:
            return True


class DARTSCell(NASBench201Cell):

    def __init__(self, n_vertices=6, vertice_type=DARTSVertice):
        # Only 6 nodes because the output node is defined as the concatenation of all other nodes.
        super().__init__(n_vertices, vertice_type)

    def to_genotype(self):
        genotype = []
        for vertice in self.vertices[2:]:
            actions = {k: v for k, v in vertice.actions.items() if v != "none"}
            for k, v in actions.items():
                genotype.append((v, k))
        return genotype

    def get_action_tuples(self):
        list_tuples = []
        for i, v in enumerate(self.vertices[2:]):
            if not v.is_complete():
                actions = v.get_action_tuples()
                for action in actions:
                    list_tuples.append((v.id, *action))
        return list_tuples

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, operation in vertice.actions.items():
                if operation is None or operation == "none":
                    op_label = 0
                else:
                    op_label = self.OPERATIONS.index(operation)
                adjacency_matrix[j,i] = op_label
        return adjacency_matrix

    def plot(self, filename="cell"):
        genotype = self.to_genotype()
        g = Digraph(
                format='pdf',
                edge_attr=dict(fontsize='20', fontname="times"),
                node_attr=dict(style='filled, rounded', shape='rect', align='center', fontsize='20', height='0.5',
                               width='0.5', penwidth='2', fontname="times"),
                engine='dot')
        g.body.extend(['rankdir=LR'])

        g.node("c_{k-2}", fillcolor='darkseagreen2')
        g.node("c_{k-1}", fillcolor='darkseagreen2')
        assert len(genotype) % 2 == 0
        steps = len(genotype) // 2

        for i in range(steps):
            g.node(str(i), fillcolor='lightblue')

        for i in range(steps):
            for k in [2 * i, 2 * i + 1]:
                op, j = genotype[k]
                if j == 0:
                    u = "c_{k-2}"
                elif j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j - 2)
                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")
        g.node("c_{k}", fillcolor='palegoldenrod')
        for i in range(steps):
            g.edge(str(i), "c_{k}", fillcolor="gray")

        g.render(filename, view=True)

class DARTSState:

    def __init__(self, state: Tuple[DARTSCell, DARTSCell]):
        self.state = state
        N_NODES = state[0].n_vertices
        self.ADJACENCY_MATRIX_SIZE = N_NODES**2
        self.N_OPERATIONS = 7
        self.zobrist_table = None

    def calculate_zobrist_hash(self, zobrist_table):
        assert zobrist_table is not None, "Remember to pass zobrist_table to node constructor."
        hash = 0
        adjacency = np.vstack([self.state[0].adjacency_matrix(), self.state[1].adjacency_matrix()]).flatten()
        for i, element in enumerate(adjacency):
            hash ^= zobrist_table[i][element]
        return hash

    def initialize_zobrist_table(self):
        self.zobrist_table = []
        for _ in range(2):  # Une fois pour la normal cell et une fois pour la reduction cell
            for i in range(ADJACENCY_MATRIX_SIZE):
                adjacency_table = []
                for operation in range(N_OPERATIONS):
                    adjacency_table.append(random.randint(0, 2 ** 64))
                self.zobrist_table.append(adjacency_table)

    def get_action_tuples(self):
        list_normal = self.state[0].get_action_tuples()
        list_normal = [(0, *e) for e in list_normal]
        list_reduction = self.state[1].get_action_tuples()
        list_reduction = [(1, *e) for e in list_reduction]

        return list(itertools.chain.from_iterable([list_normal, list_reduction]))

    def play_action(self, cell, start_vertice, end_vertice, operation):
        self.state[cell].play_action(start_vertice, end_vertice, operation)

    def is_complete(self):
        return all([s.is_complete() for s in self.state])

    def mutate(self, mutation_rate=1):
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
            normal=self.state[0].to_genotype(),
            normal_concat=[2, 3, 4, 5],
            reduce=self.state[1].to_genotype(),
            reduce_concat=[2, 3, 4, 5]
        )
        parent_compact = convert_genotype_to_compact(genotype_config)
        parent_compact = make_compact_mutable(parent_compact)
        compact = copy.deepcopy(parent_compact)
        while True:
            for _ in range(int(mutation_rate)):
                cell = np.random.choice(2)
                pair = np.random.choice(8)
                num = np.random.choice(2)
                if num == 1:
                    compact[cell][pair][num] = np.random.choice(self.N_OPERATIONS)
                else:
                    inputs = pair // 2 + 2
                    choice = np.random.choice(inputs)
                    if pair % 2 == 0 and compact[cell][pair + 1][num] != choice:
                        compact[cell][pair][num] = choice
                    elif pair % 2 != 0 and compact[cell][pair - 1][num] != choice:
                        compact[cell][pair][num] = choice

            if make_compact_immutable(parent_compact) != make_compact_immutable(compact):
                break
        new_genotype = convert_compact_to_genotype(compact)
        for i, cell_type in enumerate(["normal", "reduce"]):
            cell = eval("new_genotype." + cell_type)
            for node_idx in range(4):
                ops = cell[2*node_idx:2*node_idx+2]
                for k in range(len(self.state[i].vertices[node_idx+2].actions)):
                    for op in ops:
                        if k == op[1]:
                            self.state[i].vertices[node_idx+2].actions[k] = op[0]
                            break
                        self.state[i].vertices[node_idx + 2].actions[k] = "none"


    def get_reward(self, api, metric=Metric.VAL_ACCURACY, dataset="cifar10", df=None):
        normal_cell_genotype = self.state[0].to_genotype()
        reduction_cell_genotype = self.state[1].to_genotype()
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
                normal=normal_cell_genotype,
                normal_concat=[2, 3, 4, 5],
                reduce=reduction_cell_genotype,
                reduce_concat=[2, 3, 4, 5]
        )

        candidate = NasBench301SearchSpace()
        convert_genotype_to_naslib(genotype_config, candidate)
        reward = candidate.query(dataset=dataset, metric=metric, dataset_api=api)
        return reward

    def sample_random(self):
        while not self.is_complete():
            actions = self.get_action_tuples()
            random_action = random.choice(actions)
            self.play_action(*random_action)
