import collections
import copy
import random

import pandas as pd
from tqdm import tqdm

from monet.search_algorithms.evolutionary import EvolutionaryAlgorithm
from monet.search_algorithms.mcts_agent import MCTSAgent
from naslib.search_spaces.core import Metric
from nasbench import api as ModelSpecAPI


class RegularizedEvolution(EvolutionaryAlgorithm):

    def __init__(self, config):
        super().__init__(config)
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size
        self.population = collections.deque(maxlen=self.population_size)

    def adapt_search_space(self, search_space, dataset):
        super().adapt_search_space(search_space, dataset)
        if search_space == "nasbench101":
            self.initialize_population = self.initialize_population_101
            self.evolve = self.evolve_101

    def initialize_population(self):
        while len(self.population) < self.population_size:
            node = copy.deepcopy(self.root)
            node.sample_random()
            self.population.append(node)
            self.history.append(node)
            reward = node.get_reward(self.api, metric=Metric.VAL_ACCURACY,
                                     dataset=self.dataset, df=self.df)
            node.reward = reward
            self.all_rewards.append(reward)
            self.best_reward.append(max(self.all_rewards))

    def initialize_population_101(self):
        while len(self.population) < self.population_size:
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
            self.population.append(node)
            self.history.append(node)
            reward = node.get_reward(self.api, metric=Metric.VAL_ACCURACY,
                                     dataset=self.dataset, df=self.df)
            node.reward = reward
            self.all_rewards.append(reward)
            self.best_reward.append(max(self.all_rewards))

    def evolve(self):
        pbar = tqdm(total=self.n_iter, initial=self.population_size)
        while len(self.history) < self.n_iter:
            pbar.update(1)
            pbar.set_description(f"Current best reward : {self.best_reward[-1]:.4f}")

            samples = random.sample(self.population, self.sample_size)
            best_parent = max(samples, key=lambda x: x.reward)

            child = copy.deepcopy(best_parent)
            child.mutate()
            child.reward = child.get_reward(self.api, metric=Metric.VAL_ACCURACY,
                                            dataset=self.dataset, df=self.df)
            self.history.append(child)
            self.population.append(child)

            self.all_rewards.append(child.reward)
            self.best_reward.append(max(self.all_rewards))

        pbar.close()

    def evolve_101(self):
        pbar = tqdm(total=self.n_iter, initial=self.population_size)
        while len(self.history) < self.n_iter:
            pbar.update(1)
            pbar.set_description(f"Current best reward : {self.best_reward[-1]:.4f}")

            samples = random.sample(self.population, self.sample_size)
            best_parent = max(samples, key=lambda x: x.reward)

            child = copy.deepcopy(best_parent)
            child.mutate(api=self.api)
            child.reward = child.get_reward(self.api, metric=Metric.VAL_ACCURACY,
                                            dataset=self.dataset, df=self.df)
            self.history.append(child)
            self.population.append(child)

            self.all_rewards.append(child.reward)
            self.best_reward.append(max(self.all_rewards))

        pbar.close()

    def get_best_model(self):
        models_dict = dict(zip(self.history, self.all_rewards))
        best_model = max(models_dict, key=models_dict.get)
        best_accuracy = max(self.all_rewards)
        return best_model, best_accuracy

    def main_loop(self):
        self.initialize_population()
        self.evolve()
        best_model, best_accuracy = self.get_best_model()
        print(f"Best accuracy: {best_accuracy}")


