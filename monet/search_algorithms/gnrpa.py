import copy
import time

import numpy as np
from matplotlib import pyplot as plt

from monet.search_algorithms.nested import NRPA
from monet.utils.helpers import running_avg


class GNRPA(NRPA):

    def __init__(self, config):
        super().__init__(config)
        self.b = {}

    def adapt_search_space(self, search_space, dataset):
        super().adapt_search_space(search_space, dataset)
        self.initialize_bias()

    def initialize_bias(self):
        node_type = type(self.root)
        node = node_type(state=copy.deepcopy(self.root.state), move=None, parent=None, sequence=[])
        for action in node.get_action_tuples():
            if "conv" in str(action[-1]):
                self.b[action] = 1
            else:
                self.b[action] = 0

    def softmax_temp_fn(self, x, tau, b):
        e_x = np.exp((x / tau) + b)
        return e_x / e_x.sum()

    def adapt(self, policy, sequence):
        node_type = type(self.root)
        node = node_type(state=copy.deepcopy(self.root.state), move=None, parent=None, sequence=[])

        node.hash = node.calculate_zobrist_hash(self.root.state.zobrist_table)
        pol_prime = policy.copy()
        for action in sequence:
            code = self._code(node, action)
            if code not in pol_prime:
                # print("Erreur 0")
                pol_prime[code] = 0
            z = 0
            o = {}
            moves = node.get_action_tuples()
            for i, m in enumerate(moves):
                move_code = self._code(node, m)
                if move_code not in policy:
                #     print("Erreur 1")
                    policy[move_code] = 0
                o[m] = np.exp(policy[move_code]/self.softmax_temp + self.b[m])
                z += o[m]
            for m in moves:
                move_code = self._code(node, m)
                if move_code not in pol_prime:
                    # print("Erreur 2")
                    pol_prime[move_code] = 0
                d_bm = 1 if m == action else 0
                pol_prime[move_code] -= (self.alpha / self.softmax_temp) * ((o[m]/z) - d_bm)

            node.play_action(action)
            node.hash = node.calculate_zobrist_hash(self.root.state.zobrist_table)

        return pol_prime

class NRPALR(NRPA):

    def __init__(self, config):
        super().__init__(config)
        self.threshold = config.search.threshold
        self.i =0

    def nrpa(self, node, level, policy, alpha):

        if level == 0:
            if (len(self.rewards) - 1) % 20 == 0 and len(self.rewards) > 100:
                f, ax = plt.subplots(1, 1)
                ax.plot(running_avg(self.rewards, 10))
                f.savefig("rewards.png")
                plt.close(f)
                f, ax = plt.subplots(1, 1)
                ax.plot(running_avg(self.best_reward, 10))
                f.savefig("best_rewards.png")
                plt.close(f)
                # print(f"[{len(self.rewards)}/{self.n_iter ** self.level}] Best reward: {max(self.best_reward)}")

            score, sequence = self._playout(self.root, policy)
            # print(f"Got score {score} with sequence {sequence}")
            self.rewards.append(score)
            # print(f"[{len(self.rewards)}/{self.n_iter ** self.level}] Best reward: {max(self.best_reward)}")
            self.best_reward.append(max(self.rewards))
            self.pbar.update(1)
            self.pbar.set_description(f"Current best reward : {max(self.best_reward):.4f}, current samples {np.mean(self.rewards[-10:]):.4f}")
            self.i += 1
            return score, sequence

        else:
            # For nesting levels >= 1
            if level > 1:
                print(f"NRPA search at level {level}")
            best_score = -1
            best_sequence = []
            count_threshold = 0
            ## BREAK CONDITION
            # if len(self.rewards) > 5000:
            #     return best_score, best_sequence

            while count_threshold < self.threshold:
                if level > 1:
                    print(f"Launching a new search of level {level}")
                if self.i > self.n_iter ** self.level:
                    print("Enough iterations.")
                    return best_score, best_sequence
                t1 = time.time()
                alpha = self.alpha
                if self.lr_update:
                    alpha = self.alpha / (((self.level + 1) - (level - 1)) ** 2)
                reward, sequence = self.nrpa(node, level - 1, policy.copy(), alpha=alpha)
                t2 = time.time()
                # if level == 2:
                #     print(f"NRPA search at level {level - 1} has taken {(t2-t1):.2f} seconds")
                if set(sequence) == set(best_sequence) and level != self.level:
                    count_threshold += 1
                elif reward > best_score:
                    # print(reward, best_score, reward-best_score)
                    # print(f"best score {best_score} -> {reward}")
                    best_score = reward
                    best_sequence = sequence
                    count_threshold = 0
                # if level >= 1:
                #     print(f"[NRPA LEVEL {level}] with alpha = {alpha:.2f} Best sequence : {best_sequence}")
                policy = self.adapt(policy, best_sequence)

                # if count_threshold >= self.threshold:
                #     # print(f"[LEVEL {level}] Returning score {best_score}")
                #     return best_score, best_sequence

            return best_score, best_sequence

class GNRPALR(GNRPA, NRPALR):

    def __init__(self, config):
        super().__init__(config)

    def adapt_search_space(self, search_space, dataset):
        GNRPA.adapt_search_space(self, search_space, dataset)

    def adapt(self, policy, sequence):
        return GNRPA.adapt(self, policy, sequence)

    def nrpa(self, node, level, policy, alpha):
        return NRPALR.nrpa(self, node, level, policy, alpha)
