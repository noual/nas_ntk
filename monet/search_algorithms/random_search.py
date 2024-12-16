import copy

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from monet.search_algorithms.evolutionary import EvolutionaryAlgorithm
from monet.search_algorithms.mcts_agent import MCTSAgent
from naslib.search_spaces.core import Metric
from nasbench import api as ModelSpecAPI


class RandomSearch(EvolutionaryAlgorithm):

    def __init__(self, config):
        super().__init__(config)

    def adapt_search_space(self, search_space, dataset):
        super().adapt_search_space(search_space, dataset)
        if search_space == "nasbench101":
            self.evolve = self.evolve_101

    def evolve(self):
        for i in tqdm(range(self.n_iter)):
            node = copy.deepcopy(self.root)
            node.sample_random()
            reward = node.get_reward(self.api, metric=Metric.VAL_ACCURACY,
                                     dataset=self.dataset, df=self.df)
            self.history.append(node)
            self.all_rewards.append(reward)
            self.best_reward.append(max(self.all_rewards))

    def evolve_101(self):
        for i in tqdm(range(self.n_iter)):
            is_valid = False
            while not is_valid:
                node = copy.deepcopy(self.root)
                node.sample_random()
                adjacency, operations = node.state.operations_and_adjacency()
                model_spec = ModelSpecAPI.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=adjacency,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=operations)
                is_valid = self.api.is_valid(model_spec)

            reward = node.get_reward(self.api, metric=Metric.VAL_ACCURACY,
                                     dataset=self.dataset, df=self.df)
            self.history.append(node)
            self.all_rewards.append(reward)
            self.best_reward.append(max(self.all_rewards))

    def get_best_model(self):
        models_dict = dict(zip(self.history, self.all_rewards))
        best_model = max(models_dict, key=models_dict.get)
        best_accuracy = max(self.all_rewards)
        return best_model, best_accuracy

    def main_loop(self):
        self.evolve()
        best_model, best_accuracy = self.get_best_model()
        print(f"Best accuracy: {best_accuracy}")