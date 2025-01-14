import random

import numpy as np
from tqdm import tqdm

from monet.search_spaces.nasbench201_node import NASBench201Cell, NASBench201Vertice
from naslib.search_spaces import TransBench101SearchSpaceMacro, TransBench101SearchSpaceMicro
from naslib.search_spaces.core import Metric
from naslib.search_spaces.transbench101.conversions import convert_op_indices_macro_to_str, \
    convert_op_indices_micro_to_str
from naslib.utils.get_dataset_api import get_transbench101_api


class TransBenchVertice(NASBench201Vertice):

    def __init__(self, id):
        super().__init__(id)
        self.OPERATIONS = ["skip-connect", "none", "nor_conv_3x3", "nor_conv_1x1"]


class TransBenchCell(NASBench201Cell):

    def __init__(self, n_vertices=4):
        super().__init__(n_vertices, vertice_type=TransBenchVertice)

    def to_op_indices(self):
        edge_list = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
        ops = []
        for i, j in edge_list:
            ops.append(self.vertices[j-1].actions[i-1])
        op_names = self.vertices[0].OPERATIONS
        op_indices = [op_names.index(op) for op in ops]
        return op_indices

    def get_reward(self, api, metric=Metric.VAL_ACCURACY, dataset="jigsaw", df=None):
        task = api["task"]
        if df is not None:
            op_indices = self.to_op_indices()
            op_indices = convert_op_indices_micro_to_str(op_indices)
            return df[df["arch_str"] == op_indices][task].values[0]
        op_indices = self.to_op_indices()
        candidate = TransBench101SearchSpaceMicro()
        candidate.set_op_indices(op_indices)
        reward = candidate.query(dataset=task, metric=metric, dataset_api=api)
        # print(f"{op_indices} -> {reward}")
        return reward

    def mutate(self):
        vertice = random.choice(range(1, self.n_vertices))
        id = random.choice(range(vertice))
        op_names = self.vertices[0].OPERATIONS
        action = random.choice([op for op in op_names if op != self.vertices[vertice].actions[id]])
        self.play_action(vertice, id, action)

class TransBenchMacro:

    def __init__(self):
        self.downsamples = 0*np.ones((6))
        self.double_channels = 0*np.ones((6))
        self.depth = 0
        self.terminate = False
        self.zobrist_table = None

    def is_terminal(self):
        if self.depth == 0:
            return False
        if sum(self.downsamples > 0) < 1:
            return False
        if sum(self.double_channels > 0) < 1:
            return False
        return True

    def is_complete(self):
        if self.is_terminal():
            if self.terminate:
                return True
        return False

    def get_action_tuples(self):
        action_tuples = []
        if self.depth == 0:
            for depth in [4,5,6]:
                action_tuples.append(("set_depth", depth))
        # Reduction indices
        if sum(self.downsamples > 0) < 4:
            for i in range(max(4, self.depth)):
                if self.downsamples[i] == 0:
                    action_tuples.append(("set_downsample", i))
        if sum(self.double_channels > 0) < 3:
            for i in range(max(4, self.depth)):
                if self.double_channels[i] == 0:
                    action_tuples.append(("set_double_channels", i))
        if self.is_terminal():
            action_tuples.append(("terminate", 1))
        return action_tuples

    def play_action(self, action_name, action):
        if action_name == "set_depth":
            self.depth = action
        elif action_name == "set_downsample":
            self.downsamples[action] = 1
        elif action_name == "set_double_channels":
            self.double_channels[action] = 1
        elif action_name == "terminate":
            self.terminate = True

    def initialize_zobrist_table(self):  #UNTESTED
        self.zobrist_table = []
        for i in range(6):
            stage_table = []
            for j in range(3):
                self.zobrist_table.append(random.randint(0, 2**64))
            self.zobrist_table.append(stage_table)

    def calculate_zobrist_hash(self, zobrist_table):
        return 2

    def sample_random(self):
        while not self.is_complete():
            action_tuples = self.get_action_tuples()
            # print(action_tuples)
            action = action_tuples[np.random.randint(0, len(action_tuples))]
            # print(f"Playing action {action}")
            self.play_action(*action)

    def to_str(self):
        assert self.is_complete()
        downsamples = self.downsamples[:self.depth].astype(int)
        double_channels = self.double_channels[:self.depth].astype(int)
        op_indices = (1 + 2*downsamples + double_channels).astype(int)
        while len(op_indices) < 6:
            op_indices = np.append(op_indices, 0)
        return op_indices

    def get_reward(self, api, metric=Metric.VAL_ACCURACY, dataset="cifar10", df=None):
        task = api["task"]
        if df is not None:
            print("df")
            op_indices = self.to_str()
            op_indices = convert_op_indices_macro_to_str(op_indices)
            return df[df["arch_str"] == op_indices][task].values[0]
        op_indices = self.to_str()
        candidate = TransBench101SearchSpaceMacro()
        candidate.set_op_indices(op_indices)
        reward = candidate.query(dataset=task, metric=metric, dataset_api=api)
        # print(f"{op_indices} -> {reward}")
        return reward

    def mutate(self):

        parent_op_indices = self.to_str()
        parent_op_ind = parent_op_indices[parent_op_indices != 0]  # Mettre un np.array

        def f(g):
            r = len(g)
            p = sum([int(i == 4 or i == 3) for i in g])
            q = sum([int(i == 4 or i == 2) for i in g])
            return r, p, q

        def g(r, p, q):
            u = [2 * int(i < p) for i in range(r)]
            v = [int(i < q) for i in range(r)]
            w = [1 + sum(x) for x in zip(u, v)]
            return np.random.permutation(w)

        a, b, c = f(parent_op_ind)

        a_available = [i for i in [4, 5, 6] if i != a]
        b_available = [i for i in range(1, 5) if i != b]
        c_available = [i for i in range(1, 4) if i != c]

        dic1 = {1: a, 2: b, 3: c}
        dic2 = {1: a_available, 2: b_available, 3: c_available}

        numb = random.randint(1, 3)

        dic1[numb] = random.choice(dic2[numb])

        op_indices = g(dic1[1], dic1[2], dic1[3])
        while len(op_indices) < 6:
            op_indices = np.append(op_indices, 0)
        print(parent_op_indices)
        print(op_indices)
        raise Exception




if __name__ == '__main__':
    nets = []
    n_actions = []
    api = get_transbench101_api()
    rew = [0]
    for j in tqdm(range(1000000)):
        if j % 1000 == 0:
            print(f"Max reward : {max(rew)}")
            print("Unique nets : ",len(np.unique(nets, axis=0)))
        cell = TransBenchMacro()
        i = 0
        while not cell.is_complete():
            action_tuples = cell.get_action_tuples()
            # print(action_tuples)
            action = action_tuples[np.random.randint(0, len(action_tuples))]
            # print(f"Playing action {action}")
            cell.play_action(*action)
            i += 1
        n_actions.append(i)
        nets.append(cell.to_str())
        reward = cell.get_reward(api=api)
        rew.append(reward)
    print(len(np.unique(nets, axis=0)))
    print(f"N actions : {np.mean(n_actions)}")
    print(f"Max reward : {max(rew)}")