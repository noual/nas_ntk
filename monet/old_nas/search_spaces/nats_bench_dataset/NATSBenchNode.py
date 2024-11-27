import numpy as np
from copy import deepcopy
import random
import xautodl
from xautodl.models import get_cell_based_tiny_net

ADJACENCY_MATRIX_SIZE = 5
N_OPERATIONS = 8

class NATSBenchSizeCell:

    def __init__(self):
        self.sizes = [8, 16, 24, 32, 40, 48, 56, 64]
        self.state = {i: None for i in range(5)}
        self.ADJACENCY_MATRIX_SIZE = 5
        self.N_OPERATIONS = len(self.sizes)
        self.zobrist_table = None

    def adjacency_matrix(self):
        adj = np.array(list(self.state.values()))
        adj_indexes = []
        for a in adj:
            if a is None:
                adj_indexes.append(0)
            else:
                adj_indexes.append(self.sizes.index(a))
        return adj_indexes

    def add_zobrist(self, zobrist_table):
        ### Normalement cette fonction est seulement censée être utilisée pour la racine.
        if self.hash is None:  # Ne pas recalculer si on a déjà calculé
            self.hash = self.calculate_zobrist_hash(zobrist_table)

    def calculate_zobrist_hash(self, zobrist_table):
        assert zobrist_table is not None, "Remember to pass zobrist_table to node constructor."
        hash = 0
        adjacency = self.adjacency_matrix()
        for i, row in enumerate(adjacency):
            hash ^= zobrist_table[i][row]
        return hash

    def initialize_zobrist_table(self):
        self.zobrist_table = []
        for i in range(ADJACENCY_MATRIX_SIZE):
            adjacency_table = []
            for operation in range(N_OPERATIONS):
                adjacency_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(adjacency_table)

    def get_action_tuples(self):
        list_actions = []
        for k, v in self.state.items():
            if v is None:
                for size in self.sizes:
                    list_actions.append((k, size))
        return list_actions

    def to_str(self):
        return "{}:{}:{}:{}:{}".format(*list(self.state.values()))

    def play_action(self, key, value):
        self.state[key] = value

    def is_complete(self):
        return None not in self.state.values()

    def get_reward(self, api, df):
        if df is not None:
            arch_str = self.to_str()
            idx = api.query_index_by_arch(arch_str)
            row = df.loc[df["index"] == idx]
            reward = row["score"].item()
            ntk = row["score"].item()
            return reward
        # 1. Find the string associated to the current state
        state_str = self.to_str()
        # 2. Find the associated architecture index in the api
        index = api.query_index_by_arch(state_str)
        # 3. Fetch desired metric from API.
        reward = api.get_more_info(index, 'cifar100', hp="90")["test-accuracy"]
        # print(f"[PLAYOUT] reward = {reward}")
        return reward

    def get_metric(self, metric, api, df):
        assert metric in ["n_params"], "Metric must be one of [\"n_params\"]"
        if metric == "n_params":
            # 1. Find the string associated to the current state
            state_str = self.to_str()
            # 2. Find the associated architecture index in the api
            index = api.query_index_by_arch(state_str)
            # 3. Fetch desired metric from API.
            config = api.get_net_config(index, 'cifar10')
            network = get_cell_based_tiny_net(config)
            n_params = sum(p.numel() for p in network.parameters())
            return n_params

    def mutate(self):
        mutated = deepcopy(self)
        id = random.choice(range(5))
        size = random.choice([s for s in self.sizes if s!=self.state[id]])
        mutated.play_action((id, size))
        return mutated

    def get_neighboors(self):
        neighboors = []
        for id in range(5):
            for size in [s for s in self.sizes if s!=self.state[id]]:
                new_cell = deepcopy(self)
                new_cell.play_action((id, size))
                neighboors.append(new_cell)
        return neighboors
        