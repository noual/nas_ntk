import multiprocessing
from multiprocessing import Process

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from yacs.config import CfgNode
import os
print(os.getcwd())
import sys
sys.path.append("../..")
from monet.node import Node
from monet.search_algorithms.mcts_agent import UCT, RAVE, MCTSAgent
from monet.search_algorithms.nested import NRPA
from monet.search_spaces.nasbench201_node import NASBench201Cell
from monet.utils.helpers import configure_seaborn
from naslib.optimizers import RegularizedEvolution, Bananas
from naslib.search_spaces.core import Metric
from naslib.utils import get_dataset_api
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace

SEARCH_SPACE = "nasbench201"
DATASET = "cifar10"

def run_mcts(algorithm, config):
    alg = algorithm(config)
    alg.adapt_search_space(SEARCH_SPACE, DATASET)
    alg.main_loop()
    return alg.best_reward

def run_naslib(algorithm, config):
    alg = algorithm(config)
    api = get_dataset_api("nasbench201", 'cifar10')
    alg.adapt_search_space(NasBench201SearchSpace())
    alg.dataset_api = api
    for epoch in tqdm(range(config.search.epochs)):
        alg.new_epoch(epoch)
    return alg.best_metric

def run_once(algo_dict):
    rewards = {}
    for name, properties in algo_dict.items():
        print(f"Running {name}")
        optimizer = properties["algorithm"]
        config = properties["config"]

        if issubclass(optimizer, MCTSAgent):
            best_reward = run_mcts(optimizer, config)
        else:
            best_reward = run_naslib(optimizer, config)
        rewards[name] = best_reward
    return rewards

def run_all(algo_dict):
    print(f"Process id : {os.getpid()}")
    all_results = []
    for n_run in range(N_RUNS):

        rewards = run_once(algorithms)
        for name, reward in rewards.items():
            for iteration, score in enumerate(reward):
                all_results.append({
                    "algorithm": name,
                    "run": n_run,
                    "iteration": iteration,
                    "score": score
                })
        df = pd.DataFrame(all_results)
        df.to_csv("results.csv")

if __name__ == '__main__':
    configure_seaborn()
    N_API_CALLS = 3000
    N_RUNS = 20

    algorithms = {
        # "NRPA": {
        #     "algorithm": NRPA,
        #     "config": CfgNode.load_cfg(open('../../naslib/configs/nrpa.yaml'))
        # },
        # "UCT": {
        #     "algorithm": UCT,
        #     "config": CfgNode.load_cfg(open('../../naslib/configs/uct.yaml'))
        # },
        # "RAVE": {
        #     "algorithm": RAVE,
        #     "config": CfgNode.load_cfg(open('../../naslib/configs/uct.yaml'))
        # },
        # "RE": {
        #     "algorithm": RegularizedEvolution,
        #     "config": CfgNode({
        #         "dataset": "cifar10",
        #         "search": {
        #             "epochs": N_API_CALLS,
        #             "sample_size": 25,
        #             "population_size": 100
        #         },
        #         "df_path": "../csv/nasbench201.csv"
        #     })
        # },
        "BANANAS": {
            "algorithm": Bananas,
            "config": CfgNode({
                "search": {
                    "acq_fn_optimization": "mutation",
                    "k": 100,
                    "num_init": 10,
                    "num_ensemble": 3,
                    "predictor_type": "bananas",
                    "acq_fn_type": "its",
                    "encoding_type": "path",
                    "num_arches_to_mutate": 1,
                    "max_mutations": 1,
                    "max_candidates": 200,
                    "num_candidates": 50,
                    "epochs": 3000,
                },
                "dataset": "cifar10",
                "df_path": "../csv/nasbench201.csv"
            })
        }
    }

    if __name__ == "__main__":
        # processes = []
        # for i in range(8):  # Create 4 processes
        #     p = Process(target=run_all, args=(algorithms,))
        #     p.start()
        #     processes.append(p)
        #
        # for p in processes:
        #     p.join()
        # n = multiprocessing.cpu_count()  # guard against counting only active cores
        # with multiprocessing.Pool(n) as pool:
        #     pool.map(run_all, algorithms)


        run_all(algorithms)
    # best_reward_uct = []
    # best_reward_rave = []
    # best_reward_nrpa = []
    # best_reward_re = []
    #
    # # Run NRPA
    # algorithm = NRPA
    # config = CfgNode.load_cfg(open('../../naslib/configs/nrpa.yaml'))
    #
    # for n_run in range(N_RUNS):
    #     best_reward = run_mcts(algorithm, config)
    #     best_reward_nrpa.append(best_reward)
    # print(f"NRPA best reward : {np.mean(best_reward_nrpa, axis=0)[-1]}")
    # np.save("nrpa.npy", best_reward_nrpa)
    #
    # # Run UCT
    # algorithm = UCT
    # config = CfgNode.load_cfg(open('../../naslib/configs/uct.yaml'))
    #
    # for n_run in range(N_RUNS):
    #     best_reward = run_mcts(algorithm, config)
    #     best_reward_uct.append(best_reward)
    # print(f"UCT best reward : {np.mean(best_reward_uct, axis=0)[-1]}")
    # np.save("uct.npy", best_reward_uct)
    #
    # # Run RAVE
    # algorithm = RAVE
    # config = CfgNode.load_cfg(open('../../naslib/configs/uct.yaml'))
    #
    # for n_run in range(N_RUNS):
    #     best_reward = run_mcts(algorithm, config)
    #     best_reward_rave.append(best_reward)
    # print(f"RAVE best reward : {np.mean(best_reward_rave, axis=0)[-1]}")
    # np.save("rave.npy", best_reward_rave)
    #
    # config = CfgNode()
    # config.dataset = 'cifar10'
    # config.search = CfgNode()
    # config.search.epochs = N_API_CALLS
    # config.search.sample_size = 25
    # config.search.population_size = 100
    # api = get_dataset_api("nasbench201", 'cifar10')
    # # Run RE
    # for n_run in range(N_RUNS):
    #     re = RegularizedEvolution(config)
    #     re.adapt_search_space(NasBench201SearchSpace())
    #     re.dataset_api = api
    #     # Run regularized evolution
    #     for epoch in tqdm(range(config.search.epochs)):
    #         re.new_epoch(epoch)
    #     final_reward = re.get_final_architecture().query(dataset='cifar10', metric=Metric.VAL_ACCURACY, dataset_api=api)
    #     best_reward_re.append(re.best_metric)
    # print(f"RE best reward : {np.mean(best_reward_re, axis=0)[-1]}")
    # np.save("re.npy", best_reward_re)

    #
    # # Run BANANAS
    # config.search.acq_fn_optimization = "mutation"
    # config.search.k = 10
    # config.search.num_init = 10
    # config.search.num_ensemble = 3
    # config.search.predictor_type = "bananas"
    # config.search.acq_fn_type = "its"
    # config.search.encoding_type = "path"
    # config.search.num_arches_to_mutate = 1
    # config.search.max_mutations = 1
    # config.search.max_candidates = 200
    # config.search.num_candidates = 50
    # for n_run in range(N_RUNS):
    #     bananas = Bananas(config)
    #     bananas.adapt_search_space(NasBench201SearchSpace())
    #     bananas.dataset_api = api
    #     # Run Bananas
    #     for epoch in tqdm(range(config.search.epochs)):
    #         bananas.new_epoch(epoch)
    #     final_reward = bananas.get_final_architecture().query(dataset='cifar10', metric=Metric.VAL_ACCURACY, dataset_api=api)
    #     best_reward_bananas.append(bananas.best_metric)
    #     best_acc_bananas.append(bananas.best_metric[-1])
    #
    # print(f"RE : {np.mean(best_acc_re)}")
    # print(f"UCT : {np.mean(best_acc_uct)}")
    # print(f"BANANAS : {np.mean(best_acc_bananas)}")
    # print(f"NRPA : {np.mean(best_acc_nrpa)}")
    # #
    # f, ax = plt.subplots(1,1,figsize=(8,6))
    # ax.plot(np.mean(best_reward_uct, axis=0), label="UCT")
    # ax.fill_between(range(len(best_reward_uct[0])),
    #                 np.mean(best_reward_uct, axis=0) - np.std(best_reward_uct, axis=0),
    #                 np.mean(best_reward_uct, axis=0) + np.std(best_reward_uct, axis=0),
    #                 alpha=0.2)
    # ax.plot(np.mean(best_reward_rave, axis=0), label="RAVE")
    # ax.fill_between(range(len(best_reward_rave[0])),
    #                 np.mean(best_reward_rave, axis=0) - np.std(best_reward_rave, axis=0),
    #                 np.mean(best_reward_rave, axis=0) + np.std(best_reward_rave, axis=0),
    #                 alpha=0.2)
    # ax.plot(np.mean(best_reward_nrpa, axis=0), label="NRPA")
    # ax.fill_between(range(len(best_reward_nrpa[0])),
    #                 np.mean(best_reward_nrpa, axis=0) - np.std(best_reward_nrpa, axis=0),
    #                 np.mean(best_reward_nrpa, axis=0) + np.std(best_reward_nrpa, axis=0),
    #                 alpha=0.2)
    # ax.plot(np.mean(best_reward_re, axis=0), label="Regularized Evolution")
    # ax.fill_between(range(len(best_reward_re[0])),
    #                 np.mean(best_reward_re, axis=0) - np.std(best_reward_re, axis=0),
    #                 np.mean(best_reward_re, axis=0) + np.std(best_reward_re, axis=0),
    #                 alpha=0.2)
    # # ax.plot(np.mean(best_reward_bananas, axis=0), label="BANANAS")
    # # ax.fill_between(range(len(best_reward_bananas[0])),
    # #                 np.mean(best_reward_bananas, axis=0) - np.std(best_reward_bananas, axis=0),
    # #                 np.mean(best_reward_bananas, axis=0) + np.std(best_reward_bananas, axis=0),
    # #                 alpha=0.2)
    #
    # ax.set_ylabel("Accuracy"); ax.set_xlabel("# of network evaluations")
    # f.suptitle("NAS-Bench-201")
    # ax.legend()
    # plt.ylim([88, 92])
    # plt.tight_layout()
    # plt.show()
