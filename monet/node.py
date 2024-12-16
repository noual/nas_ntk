from naslib.search_spaces.core import Metric


class Node:
    """
    Représente un noeud dans l'arbre de recherche.
    """

    def __init__(self, state, move=None, parent=None, sequence=[], zobrist_table=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.results = []
        self.amaf = []  # For RAVE and variants
        self.sequence = sequence  # For Nested MCS and variants
        self.hash = None
        if zobrist_table is not None:
            self.hash = self.calculate_zobrist_hash(zobrist_table)

    def get_children(self):
        return self.children

    def is_leaf(self):
        return True if len(self.children) == 0 else False

    def is_terminal(self):
        return self.state.is_complete()

    def get_action_tuples(self):
        return self.state.get_action_tuples()

    def calculate_zobrist_hash(self, zobrist_table):
        return self.state.calculate_zobrist_hash(zobrist_table)

    def add_zobrist(self, zobrist_table):
        ### Normalement cette fonction est seulement censée être utilisée pour la racine.
        if self.hash is None:  # Ne pas recalculer si on a déjà calculé
            self.hash = self.calculate_zobrist_hash(zobrist_table)

    def get_reward(self, api, metric=Metric.VAL_ACCURACY, dataset="cifar10", df=None):
        return self.state.get_reward(api, metric, dataset, df)

    def play_action(self, action):
        self.state.play_action(*action)

    def has_predecessor(self, node):
        """
        Return True if self is a child of node (can be several generations)
        :param node:
        :return:
        """
        temp_parent = self.parent
        while temp_parent.parent is not None:
            if temp_parent == node:
                return True
            temp_parent = temp_parent.parent
        if temp_parent == node:  # Include root (pas sûr)
            return True
        return False

    def sample_random(self):
        return self.state.sample_random()

    def mutate(self, **kwargs):
        return self.state.mutate(**kwargs)