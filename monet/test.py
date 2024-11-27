import random

from yacs.config import CfgNode

import naslib as nl
from naslib.search_spaces import NasBench201SearchSpace
from naslib.optimizers import RandomSearch
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric

# Set up the search space and dataset API
search_space = NasBench201SearchSpace()
dataset_api = get_dataset_api("nasbench201", 'cifar10')

# Initialize the random search optimizer
config = CfgNode()
config.dataset = 'cifar10'
config.search = CfgNode()
config.search.fidelity = 100

optimizer = RandomSearch(config=config)
optimizer.adapt_search_space(search_space)

# Perform the random search
num_evaluations = 50
best_accuracy = 0.0
best_architecture = None

for _ in range(num_evaluations):
    # Sample a random architecture
    search_space = NasBench201SearchSpace()
    search_space.sample_random_architecture()

    # Evaluate the architecture

    results = search_space.query(dataset='cifar10', metric=Metric.VAL_ACCURACY, dataset_api=dataset_api)
    accuracy = results

    # Update the best architecture if the current one is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_architecture = search_space.clone()

# Output the best architecture and its accuracy
print(f"Best architecture: {best_architecture}")
print(f"Best accuracy: {best_accuracy:.2f}%")
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str

# Assuming `search_space` is an instance of NasBench201SearchSpace
arch_str = convert_naslib_to_str(best_architecture)
print(f"Architecture string: {arch_str}")