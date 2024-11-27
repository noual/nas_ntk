import numpy as np
import torch
from torch import nn
# Works for NASBench201 API -- Some modifications for Basic Model
from torch._functorch.make_functional import make_functional, make_functional_with_buffers
from torch.func import vmap, jacrev, jacfwd, functional_call, vjp, jvp


class Scalar_NN(nn.Module):
    def __init__(self, network, class_val=2):
        super(Scalar_NN, self).__init__()
        self.network = network
        self.class_val = class_val

    def forward(self, x):
        if len(self.network(x)) > 1:
            return self.network(x)[-1][:, self.class_val]
        else:
            return self.network(x)[:, self.class_val]


def model_min_eigenvalue_class(model, x, class_val):

    model = Scalar_NN(network=model, class_val=class_val)

    def fnet_single(params, x):
        return functional_call(model, params, x.unsqueeze(0))[-1].squeeze(0)

    parameters = {k: v.detach() for k, v in model.named_parameters()}

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False

    jac1 = vmap(jacrev(fnet_single), (None, 0))(parameters, x)
    jac1 = tuple(jac1[name] for name in jac1)
    jac1 = [j.flatten(1) for j in jac1]

    jac2 = jac1

    result = torch.stack([torch.einsum( 'Na,Mb->NM', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    u, sigma, v = torch.linalg.svd(result)

    return torch.min(sigma)


def compute_score(model, dataset_classes, class_permutation, device="cpu"):
    model = model.to(device)
    lambdas = []
    for c in dataset_classes.keys():
        x_ntks = dataset_classes[c]
        if class_permutation is not None:
            c = class_permutation[c]
        lam = model_min_eigenvalue_class(model, x_ntks, c)
        lambdas.append(lam.cpu().numpy())
    return np.mean(lambdas)