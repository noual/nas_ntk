import argparse
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
from monet.search_spaces.nasbench101_node import NASBench101Cell
from monet.utils.helpers import configure_seaborn
from naslib.optimizers import RegularizedEvolution, Bananas, RandomSearch
from naslib.search_spaces.core import Metric
from naslib.utils import get_dataset_api
from naslib.search_spaces.nasbench101.graph import NasBench101SearchSpace

SEARCH_SPACE = "nasbench101"
DATASET = "cifar10"
N_ITER = 3500

def run_mcts(algorithm, config):
    config.search.n_iter = N_ITER
    alg = algorithm(config)
    alg.adapt_search_space(SEARCH_SPACE, DATASET)
    alg.main_loop()
    return alg.best_reward

def run_naslib(algorithm, config):
    config.search.epochs = N_ITER
    alg = algorithm(config)
    api = get_dataset_api("nasbench101", 'cifar10')
    alg.adapt_search_space(NasBench101SearchSpace())
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

def run_all(algo_dict, output_file="results_local"):
    print(algo_dict == algorithms)
    print(f"Process id : {os.getpid()}")
    all_results = []
    for n_run in range(N_RUNS):

        rewards = run_once(algo_dict)
        for name, reward in rewards.items():
            for iteration, score in enumerate(reward):
                all_results.append({
                    "algorithm": name,
                    "run": n_run,
                    "iteration": iteration,
                    "score": score
                })
        df = pd.DataFrame(all_results)
        df.to_csv(f"nasbench101_{output_file}.csv")

if __name__ == '__main__':
    configure_seaborn()
    N_API_CALLS = CfgNode.load_cfg(open('../../naslib/configs/nrpa.yaml')).search.n_iter
    N_RUNS = 200

    algorithms = {
        # "NRPA_L1": {
        #     "algorithm": NRPA,
        #     "config": CfgNode({
        #         "df_path": "../../monet/csv/nasbench101.csv",
        #         "dataset": "cifar10",
        #         "search": {
        #             "level": 1,
        #             "nrpa_alpha": 0.1,
        #             "softmax_temp": 2,
        #             "playouts_per_selection": 1,
        #             "C": 0.1,
        #             "n_iter": 2200,
        #             "rave_b": 0.1,
        #         },
        #         "disable_tqdm": "true",
        #         "seed": 0
        #     })
        # },
        # "NRPA_L2": {
        #     "algorithm": NRPA,
        #     "config": CfgNode({
        #         "df_path": "../../monet/csv/nasbench101.csv",
        #         "dataset": "cifar10",
        #         "search": {
        #             "level": 2,
        #             "nrpa_alpha": 0.1,
        #             "softmax_temp": 2,
        #             "playouts_per_selection": 1,
        #             "C": 0.1,
        #             "n_iter": 2200,
        #             "rave_b": 0.1,
        #         },
        #         "disable_tqdm": "true",
        #         "seed": 0
        #     })
        # },
        "NRPA_L3": {
            "algorithm": NRPA,
            "config": CfgNode({
                "df_path": "../../monet/csv/nasbench101.csv",
                "dataset": "cifar10",
                "search": {
                    "level": 3,
                    "nrpa_alpha": 0.1,
                    "softmax_temp": 2,
                    "playouts_per_selection": 1,
                    "C": 0.1,
                    "n_iter": 2200,
                    "rave_b": 0.1,
                },
                "disable_tqdm": "true",
                "seed": 0
            })
        },
        "RS": {
            "algorithm": RandomSearch,
            "config": CfgNode({
                "dataset": "cifar10",
                "search": {
                    "epochs": 200,
                    "fidelity": 1
                },
                "df_path": "../../monet/csv/nasbench101.csv"
            })
        },
        "UCT": {
            "algorithm": UCT,
            "config": CfgNode({
                "df_path": "../../monet/csv/nasbench101.csv",
                "dataset": "cifar10",
                "search": {
                    "playouts_per_selection": 1,
                    "C": 0.1,
                    "n_iter": 2200,
                    "rave_b": 0.1},
                "disable_tqdm": False,
                "seed": 0,

            })
        },
        "RAVE": {
            "algorithm": RAVE,
            "config": CfgNode({
                "df_path": "../../monet/csv/nasbench101.csv",
                "dataset": "cifar10",
                "search": {
                    "playouts_per_selection": 1,
                    "C": 0.1,
                    "n_iter": 2200,
                    "rave_b": 0.1},
                "disable_tqdm": False,
                "seed": 0,

            })
        },
        "RE": {
            "algorithm": RegularizedEvolution,
            "config": CfgNode({
                "dataset": "cifar10",
                "df_path": "../../monet/csv/nasbench101.csv",
                "search": {
                    "epochs": N_API_CALLS,
                    "sample_size": 25,
                    "population_size": 100
                },
            })
        }
    }

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Run NAS algorithms")
        parser.add_argument("--output_file", type=str, default="results_local", help="Output file for results")
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

        run_all(algorithms, output_file=parser.parse_args().output_file)
