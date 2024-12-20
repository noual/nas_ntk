import argparse
import multiprocessing
from multiprocessing import Process

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from yacs.config import CfgNode
import os

from monet.search_algorithms.evolutionary import EvolutionaryAlgorithm

print(os.getcwd())
import sys
sys.path.append("../..")
from monet.node import Node
from monet.search_algorithms.mcts_agent import UCT, RAVE, MCTSAgent
from monet.search_algorithms.regularized_evolution import RegularizedEvolution as MonetRegularizedEvolution
from monet.search_algorithms.random_search import RandomSearch as MonetRandomSearch
from monet.search_algorithms.nested import NRPA, BeamNRPA
from monet.search_spaces.nasbench101_node import NASBench101Cell
from monet.utils.helpers import configure_seaborn
from naslib.optimizers import RegularizedEvolution, Bananas, RandomSearch
from naslib.search_spaces.core import Metric
from naslib.utils import get_dataset_api
from naslib.search_spaces.nasbench301.graph import NasBench301SearchSpace

SEARCH_SPACE = "nasbench301"
DATASET = "cifar10"
N_ITER = 10000

def run_mcts(algorithm, config):
    config.search.n_iter = N_ITER
    alg = algorithm(config)
    alg.adapt_search_space(SEARCH_SPACE, DATASET)
    alg.main_loop()
    return alg.best_reward

def run_naslib(algorithm, config):
    config.search.epochs = N_ITER
    alg = algorithm(config)
    api = get_dataset_api(SEARCH_SPACE, DATASET)
    alg.adapt_search_space(NasBench301SearchSpace())
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

        if issubclass(optimizer, MCTSAgent) or issubclass(optimizer, EvolutionaryAlgorithm):
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
        df.to_csv(f"nasbench301_{output_file}.csv")

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
        #             "nrpa_alpha": 1,
        #             "softmax_temp": 1,
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
        #             "nrpa_alpha": 1,
        #             "softmax_temp": 1,
        #             "playouts_per_selection": 1,
        #             "C": 0.1,
        #             "n_iter": 2200,
        #             "rave_b": 0.1,
        #         },
        #         "disable_tqdm": "true",
        #         "seed": 0
        #     })
        # },
        # "NRPA_L3": {
        #     "algorithm": NRPA,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "dataset": "cifar10",
        #         "search": {
        #             "level": 3,
        #             "nrpa_alpha": .5,
        #             "softmax_temp": 1,
        #             "playouts_per_selection": 1,
        #             "C": 0.1,
        #             "n_iter": 2200,
        #             "rave_b": 0.1,
        #         },
        #         "disable_tqdm": "true",
        #         "seed": 0
        #     })
        # },
        # "NRPA_L3-2": {
        #     "algorithm": NRPA,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "dataset": "cifar10",
        #         "search": {
        #             "level": 3,
        #             "nrpa_alpha": 1,
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
            "algorithm": BeamNRPA,
            "config": CfgNode({
                "df_path": "none",
                "dataset": "cifar10",
                "search": {
                    "level": 4,
                    "nrpa_alpha": 0.8,
                    "softmax_temp": 1,
                    "playouts_per_selection": 1,
                    "C": 0.1,
                    "n_iter": 2200,
                    "rave_b": 0.1,
                    "beam_size": 5
                },
                "disable_tqdm": "true",
                "seed": 0
            })
        },
        "RS": {
            "algorithm": MonetRandomSearch,
            "config": CfgNode({
                "dataset": "cifar10",
                "search": {
                    "n_iter": N_API_CALLS,
                    "epochs": 200,
                    "fidelity": 1
                },
                "df_path": "none"
            })
        },
        "UCT": {
            "algorithm": UCT,
            "config": CfgNode({
                "df_path": "none",
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
                "df_path": "none",
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
            "algorithm": MonetRegularizedEvolution,
            "config": CfgNode({
                "dataset": "cifar10",
                "df_path": "none",
                "search": {
                    "n_iter": N_API_CALLS,
                    "epochs": N_API_CALLS,
                    "sample_size": 25,
                    "population_size": 100
                },
            })

         }
    }

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
