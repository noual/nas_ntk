import copy
import json
import sys

import pandas as pd
from torch import nn
from yacs.config import CfgNode

from monet.node import Node
from monet.search_spaces.nasbench101_node import NASBench101Cell
from monet.search_spaces.nasbench201_node import NASBench201Cell
from monet.search_spaces.nasbench301_node import DARTSState, DARTSCell
from naslib.search_spaces.core import Metric
from naslib.utils import get_dataset_api

sys.path.append("..")

import time
import numpy as np
from tqdm import tqdm


class MCTSAgent:

    def __init__(self, config: CfgNode()):
        self.root = None
        self.api = None
        self.df = None
        if config.df_path != "none":
            self.df = pd.read_csv(config.df_path)
        self.playouts_per_selection = config.search.playouts_per_selection
        self.C = config.search.C
        self.n_iter = config.search.n_iter
        self.disable_tqdm = config.disable_tqdm

    def adapt_search_space(self, search_space, dataset):
        print(search_space, dataset)
        assert search_space in ["nasbench201", "nasbench101", "nasbench301"], "Only NASBench301, NASBench201, NASBench101 are supported"
        if search_space == "nasbench201":
            if isinstance(self, UCT):
                print(f"Reducing number of iterations")
                self.n_iter = self.n_iter // 6
            assert dataset in ["cifar10"], "Only CIFAR10 is supported"
            self.root = Node(state=NASBench201Cell())
            self.api = get_dataset_api(search_space, dataset)

        elif search_space == "nasbench101":
            if isinstance(self, UCT):
                print(f"Reducing number of iterations")
                self.n_iter = self.n_iter // 12
            assert dataset in ["cifar10"], "Only CIFAR10 is supported"
            self.root = Node(state=NASBench101Cell(7))
            self.api = get_dataset_api(search_space, dataset)["nb101_data"]

        elif search_space == "nasbench301":
            if isinstance(self, UCT):
                print(f"Reducing number of iterations")
                self.n_iter = self.n_iter // 16
            assert dataset in ["cifar10"], "Only CIFAR10 is supported"
            self.root = Node(state= DARTSState((DARTSCell(),
                                                DARTSCell())
                                               )
                             )
            self.api = get_dataset_api(search_space, dataset)

    def _score_node(self, node: Node, parent: Node):
        pass

    def _get_reward(self, node: Node):
        pass

    def _create_network(self, node: Node) -> nn.Module:
        """
        Créer un réseau de neurones à partir du noeud MCTS
        :param node:
        :return: nn.Module
        """
        pass

    def _selection(self, node: Node):
        pass

    def _expansion(self, node: Node):
        pass

    def _playout(self, node: Node):
        pass

    def _backpropagation(self, node: Node, result: float):
        pass

    def __call__(self):
        pass


class UCT(MCTSAgent):

    def __init__(self, config:CfgNode):
        super().__init__(config)

    def _score_node(self, child: Node, parent: Node, C=None) -> float:
        # Returns UCB score for a child node
        if len(child.results) == 0:  # Si le noeud n'a pas encore été visité !
            return np.inf
        if C is None:
            C = self.C

        mu_i = np.mean(child.results)
        # print(f"[UCB] : move : {child.move}, mu_i = {mu_i}, autre param: {C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))}")
        return mu_i + C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))

    def _selection(self, node: Node) -> Node:
        """
        Selects a candidate child node from the input node.
        """
        if not node.is_leaf():  # Tant que l'on a pas atteint une feuille de l'arbre
            # Choisir le meilleur noeud enfant
            # print(f"[SELECTION] nb of candidates : {len(node.get_children())}")
            # C dynamique
            C_dif = np.max([np.nan_to_num(np.mean(c.results)) for c in node.get_children()]) - np.min(
                [np.nan_to_num(np.mean(c.results)) for c in node.get_children()])
            # print(f"[SELECTION] C_dif = {C_dif}")
            C = max([self.C, 1 * C_dif])
            # print(f"[SELECTION] Exploration constant C : {C}")
            scores = [self._score_node(child, node, C) for child in node.get_children()]
            candidate_id = np.random.choice(np.flatnonzero(scores == np.max(scores)))  # Argmax with random tie-breaks
            candidate = node.get_children()[candidate_id]
            # print(f"[SELECTION] Choosing child {candidate.move} with ucb {self._score_node(candidate, node, C)}")
            return self._selection(candidate)

        # self.current_game_moves.append(node.move)
        return node

    def _expansion(self, node: Node) -> Node:
        """
        Unless L ends the game decisively (e.g. win/loss/draw) for either player,
        create one (or more) child nodes and choose node C from one of them.
        Child nodes are any valid moves from the game position defined by L.
        """
        if not node.is_terminal():
            """
            Si le noeud n'a pas encore été exploré : on le retourne directement
            """
            if len(node.results) == 0 and node.parent is not None:
                return node
            node_type = type(node)
            node.children = [node_type(copy.deepcopy(node.state),
                                       move=m,
                                       parent=node)
                             for m in node.get_action_tuples()]
            # pprint(node.get_action_tuples())

            # Play the move for each child (updates the board in the child nodes)
            for child in node.children:
                child.play_action(child.move)
            returned_node = node.children[np.random.randint(0, len(node.children))]
            # print(f"[EXPANSION] returning random child : {returned_node.move}")
            return returned_node

        return node

    def _get_reward(self, node, metric=Metric.VAL_ACCURACY, dataset="cifar10"):
        return node.get_reward(self.api, metric, dataset, self.df)

    def _playout(self, node: Node):
        """
        Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
        :return:
        """
        node_type = type(node)
        playout_node = node_type(state=copy.deepcopy(node.state))

        while not playout_node.is_terminal():
            available_actions = playout_node.get_action_tuples()
            random_action = available_actions[np.random.randint(len(available_actions))]
            playout_node.play_action(random_action)
            # print(f"[PLAYOUT] Playing random action {random_action}")

        reward = self._get_reward(playout_node)

        del playout_node
        return reward

    def _backpropagation(self, node: Node, result: float):
        """
        Backpropagates the result of a playout up the tree.
        """
        if node.parent is None:
            node.results.append(result)
            return "Done"
        node.results.append(result)  # Ajouter le résultat à la liste
        return self._backpropagation(node.parent, result)  # Fonction récursive

    def next_best_move(self, all_rewards=None, best_reward=None) -> Node:
        """
        Body of UCT
        """
        best_reward_value = np.max(best_reward) if len(best_reward) > 0 else 0
        for i in tqdm(range(self.n_iter), disable=self.disable_tqdm):

            leaf_node = self._selection(self.root)
            expanded_node = self._expansion(leaf_node)

            for i_playout in range(self.playouts_per_selection):
                result = self._playout(expanded_node)
                _ = self._backpropagation(expanded_node, result)
                all_rewards.append(result)
                if result > best_reward_value:
                    best_reward_value = result
                best_reward.append(best_reward_value)

        best_move_id = np.argmax([np.mean(child.results) for child in self.root.get_children()])
        best_move = self.root.get_children()[best_move_id]
        # print(f"[BODY] Selecting best move {best_move.move} with mean result {np.mean(best_move.results)}")

        return best_move, all_rewards, best_reward

    def main_loop(self):
        """
        Corps de l'algorithme. Cherche le meilleur prochain coup jusqu'à avoir atteint un état terminal.
        :return: Le noeud représentant les meilleurs coups.
        """
        """Enregistrer les paramètres de la simul ation dans le folder"""
        # if self.save_folder is not None:
        #     shutil.copyfile(self.params_path, f"runs/{self.save_folder}/{self.__class__.__name__}-params.json")
        node = self.root
        self.all_rewards = []
        self.best_reward = []
        self.best_reward_value = 0

        while not node.is_terminal():
            best_move, self.all_rewards, self.best_reward = self.next_best_move(self.all_rewards, self.best_reward)
            print(best_move.move)

            node.play_action(best_move.move)
            root_type = type(self.root)
            self.root = best_move
            # print(len(best_move.children))
            # print([(len(e.results), np.mean(e.results)) for e in best_move.children])
            # self.root = root_type(copy.deepcopy(node.state))

class RAVE(UCT):

    def __init__(self, config: CfgNode):
        self.list_nodes = []
        self.b = config.search.rave_b
        super(RAVE, self).__init__(config)

    def adapt_search_space(self, search_space, dataset):
        super(RAVE, self).adapt_search_space(search_space, dataset)
        self.list_nodes.append(self.root)

    def beta(self, ni, ni_tilda):
        """
        D'après Gelly et Silver, beta(ni, ni_tilda) = n_tilda / (ni + ni_tilda + 4b^2ni*ni_tilda)
        :param ni: Nombre de parties du noeud i
        :param ni_tilda: Nombre de parties contenant le noeud i
        :return:
        """
        p = ni_tilda
        d = ni + ni_tilda + (4 * np.power(self.b, 2) * ni * ni_tilda)
        return p / d

    def _score_node(self, child: Node, parent: Node, C=None) -> int:
        # Returns UCB score for a child node
        if len(child.results) == 0:  # Si le noeud n'a pas encore été visité !
            return np.inf
        if C is None:
            C = self.C
        mu_i = np.mean(child.results)
        mu_i_tilda = np.nan_to_num(np.mean(child.amaf), 0)
        beta = self.beta(ni=len(child.results),
                         ni_tilda=len(child.amaf))
        exploration_term = C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))
        # print(f"[UCB RAVE] : move : {child.move}, mu_i = {mu_i}, mu_i_tilda = {mu_i_tilda}, beta = {beta}, autre param: {C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))}")

        return (1 - beta) * mu_i + beta * mu_i_tilda + exploration_term

    """
    Pas besoin de redéfinir la méthode de sélection
    """

    def _expansion(self, node: Node) -> Node:
        """
        Unless L ends the game decisively (e.g. win/loss/draw) for either player,
        create one (or more) child nodes and choose node C from one of them.
        Child nodes are any valid moves from the game position defined by L.
        """
        if not node.is_terminal():
            """
            Si le noeud n'a pas encore été exploré : on le retourne directement
            """
            if len(node.results) == 0 and node.parent is not None:
                return node
            node_type = type(node)
            node.children = [node_type(copy.deepcopy(node.state),
                                       move=m,
                                       parent=node)
                             for m in node.get_action_tuples()]
            # pprint(node.get_action_tuples())

            # Play the move for each child (updates the board in the child nodes)
            for child in node.children:
                self.list_nodes.append(child)
                child.play_action(child.move)
            returned_node = node.children[np.random.randint(0, len(node.children))]
            # print(f"[EXPANSION] returning random child : {returned_node.move}")
            return returned_node

        return node

    """
    Pas besoin de redéfinir la méthode de playout
    """

    def _backpropagation(self, node: Node, result: float):
        """
        Backpropagates the result of a playout up the tree.
        Also backpropagates the AMAF results.
        :param node: Current node
        :param result: Result of the playout
        :return:
        """
        if node.parent is None:
            node.results.append(result)
            return "Done"

        node.results.append(result)  # Ajouter le résultat à la liste
        for temp_node in self.list_nodes:
            if node.move == temp_node.move :#and temp_node.has_predecessor(node):
                temp_node.amaf.append(result)
        return self._backpropagation(node.parent, result)  # Fonction récursive

