import numpy as np
import torch
from torch import nn
from functorch import vmap, jacrev, make_functional, make_functional_with_buffers


class Scalar_NN(nn.Module):
    def __init__(self, network, class_val):
        super(Scalar_NN, self).__init__()
        self.network = network
        self.class_val = class_val

    def forward(self, x):
        return self.network(x)[:, self.class_val].reshape(-1, 1)

def get_jacobian(model, x, class_val):

    model = Scalar_NN(network=model, class_val=class_val)

    def fnet_single(x):
        fmodel, params, buffers =  make_functional_with_buffers(model)
        return fmodel(params, buffers, x.unsqueeze(0)).squeeze(0)

    parameters = {k: v.detach() for k, v in model.named_parameters()}

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False

    jac1 = vmap(jacrev(fnet_single) )(x)
    # jac1 = jac1.values()
    jac1 = [j.flatten(1) for j in jac1]

    jac1 = torch.cat(jac1, dim=1)
    return jac1

def compute_score(model, dataset, device="cuda"):
    samples=10
    lambda_min = []
    lambda_max = []
    for i in range(samples):
        dataset_classes = subset_classes(dataset, device=device)
        model = model.to(device)
        jacs = []
        for c in dataset_classes.keys():
            x_ntks = dataset_classes[c].to(device)
            jac = get_jacobian(model, x_ntks, c)
            jacs.append(jac)
        jacs = torch.cat(jacs, dim=0)
        ntk = torch.einsum("Na,bM->NM", jacs, jacs.T)
        # ntk = torch.dot(jacs, jacs.T)
        u, sigma, v = torch.linalg.svd(ntk)
        print(sigma)
        lambda_min.append(torch.min(sigma).item())
        lambda_max.append(torch.max(sigma).item())
    return np.mean(lambda_min), np.mean(lambda_max), ntk, dataset_classes.keys()


def subset_classes(dataset, samples_per_class=10, device="cuda"):
    dataset_classes = {}
    count_per_class = {}

    while True:
        inp, tar = dataset[np.random.randint(len(dataset))]
        try:
            if tar not in dataset_classes:
                dataset_classes[tar] = []
                count_per_class[tar] = 0
            if count_per_class[tar] < samples_per_class:
                dataset_classes[tar].append(inp.to(device))
                count_per_class[tar] += 1
        except    Exception as e:
            print(f"Error with target {tar} : {e}")

        if all(count >= samples_per_class for count in count_per_class.values()):
            break

    for key in dataset_classes.keys():
        dataset_classes[key] = torch.stack(dataset_classes[key])

    return dataset_classes