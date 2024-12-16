import pandas as pd

from monet.search_algorithms.mcts_agent import MCTSAgent


class EvolutionaryAlgorithm:
    def __init__(self, config):
        self.root = None
        self.api = None
        self.df = None
        if config.df_path != "none":
            self.df = pd.read_csv(config.df_path)
        self.n_iter = config.search.n_iter
        self.history = []
        self.all_rewards = []
        self.best_reward = []


    def adapt_search_space(self, search_space, dataset):
        MCTSAgent.adapt_search_space(self, search_space, dataset)
