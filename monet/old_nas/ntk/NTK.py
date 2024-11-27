from typing import Tuple, List

import torch
import torch.nn as nn
import tensorly as tl
tl.set_backend("pytorch")
from torch._functorch.make_functional import make_functional, make_functional_with_buffers
from torch.func import vmap, jacrev, jacfwd, functional_call, vjp, jvp

class NTK:
    """
    Compute the Neural Tangent Kernel (NTK) for a given neural network model.
    """

    def __init__(self, model, compute="full", same_sample=True, nas_api=True):
        """
        Initialize the NTK with a neural network model.

        Args:
        - model (nn.Module): Neural network model.
        - compute (str): Type of NTK computation ("full", "trace", "diagonal").
        - same_sample (bool): Whether to recompute the Jacobian if same sample.
        - nas_api(bool) : Whether the model comes from the AutoDL library
        """
        self.model = model
        self.parameters = {k: v.detach() for k, v in model.named_parameters()}
        self.same_sample = same_sample
        self.nas_api = nas_api
        if compute == "full":
            self.expr = 'Naf,Mbf->NMab'
        elif compute == "trace":
            self.expr = "Naf,Maf->NM"
        elif compute == "diagonal":
            self.expr = "Naf,Maf->NMa"

        self._disable_batchnorm_stats_tracking()

    def _disable_batchnorm_stats_tracking(self):
        """
        Disable tracking of running statistics for BatchNorm layers in the model.
        """
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False

    def fnet_single(self, params, x):
        """
        Forward pass function for a single input.

        Args:
        - params (Dict[str, Any]): Model parameters.
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor from the model.
        """
        if not self.nas_api:
            return functional_call(self.model, params, x.unsqueeze(0)).squeeze(0)
        else :
            return functional_call(self.model, params, x.unsqueeze(0))[-1].squeeze(0)

    def empirical_ntk_jacobian_contraction(self, fnet, params, x, y):
        """
        Compute the empirical NTK (Neural Tangent Kernel) via Jacobian contraction.

        Args:
        - fnet (callable): Forward pass function.
        - params (Dict[str, Any]): Model parameters.
        - x (torch.Tensor): Input tensor x.
        - y (torch.Tensor): Input tensor y.

        Returns:
        - torch.Tensor: Empirical NTK.
        """
        jac1 = vmap(jacrev(fnet), (None, 0))(params, x)
        jac1 = tuple(jac1[name] for name in jac1)
        jac1 = [j.flatten(2) for j in jac1]

        if self.same_sample:
            jac2 = jac1
        else:
            jac2 = vmap(jacrev(fnet), (None, 0))(params, y)
            jac2 = tuple(jac2[name] for name in jac2)
            jac2 = [j.flatten(2) for j in jac2]

        result = torch.stack([torch.einsum(self.expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result

    def empirical_ntk_implicit(self, x1, x2, compute="full"):
        cs=1
        fnet, params, buffers = make_functional_with_buffers(self.model)
        def fnet_single(params, x):
            return torch.flatten(fnet(params, buffers, x.unsqueeze(0)).squeeze(0))
        def get_ntk(x1, x2):
            def push_fnet_x1(params):
                return fnet_single(params, x1)
            def push_fnet_x2(params):
                return fnet_single(params, x2)
            output, vjp_fn = vjp(push_fnet_x1, params)
            def get_ntk_slice(vec):
                # Computes I @ J(x2).T
                vjps = vjp_fn(vec)
                # Computes J(x1) @ vjps
                _, jvps = jvp(push_fnet_x2, (params,), vjps)
                return jvps

            basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device)
            return vmap(get_ntk_slice, chunk_size=cs)(basis)

        result = vmap(vmap(get_ntk, (None, 0), chunk_size=cs), (0, None), chunk_size=cs)(x1, x2)
        return result


    def compute_ntk(self, x, y):
        """
        Compute the NTK (Neural Tangent Kernel).

        Args:
        - x (torch.Tensor): Input tensor x.
        - y (torch.Tensor): Input tensor y.

        Returns:
        - torch.Tensor: Computed NTK.
        """
        # return self.empirical_ntk_implicit(x1=x, x2=y)
        return self.empirical_ntk_jacobian_contraction(self.fnet_single, params=self.parameters, x=x, y=y)

    def compute_jac(self, x):
        """
        Compute the Jacobian wrt to a batch of samples.

        Args:
        - x (torch.Tensor): Input tensor x.

        Returns:
        - torch.Tensor: Computed Jacobian.
        """
        return vmap(jacrev(self.fnet_single, argnums=1), (None, 0))(self.parameters, x)


    # vmap : function that can handle batch
    # jacrev : compute the jacobian 
    # the combination of both permits to compute jacobian on batch of data

def truncated_svd(
    A: torch.Tensor,
    k: int,
    n_iter: int = 2,
    n_oversamples: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Truncated SVD.

    Based on fbpca's version.

    Parameters
    ----------
    A : (M, N) torch.Tensor
    k : int
    n_iter : int
    n_oversamples : int

    Returns
    -------
    u : (M, k) torch.Tensor
    s : (k,) torch.Tensor
    vt : (k, N) torch.Tensor
    """
    m, n = A.shape
    Q = torch.randn(n, k + n_oversamples).to(A.device)
    Q = A @ Q

    Q, _ = torch.linalg.qr(Q)

    # Power iterations
    for _ in range(n_iter):
        Q = (Q.t() @ A).t()
        Q, _ = torch.linalg.qr(Q)
        Q = A @ Q
        Q, _ = torch.linalg.qr(Q)

    QA = Q.t() @ A
    # Transpose QA to make it tall-skinny as MAGMA has optimisations for this
    # (USVt)t = VStUt
    Va, s, R = torch.linalg.svd(QA.t(), full_matrices=False)
    U = Q @ R.t()

    return U[:, :k], s[:k], Va.t()[:k, :]

def sthosvd(
        tensor: torch.Tensor,
        core_size: List[int]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Seqeuntially Truncated Higher Order SVD.

    Parameters
    ----------
    tensor : torch.Tensor,
        arbitrarily dimensional tensor
    core_size : list of int

    Returns
    -------
    torch.Tensor
        core tensor
    List[torch.Tensor]
        list of singular vectors
    List[torch.Tensor]
        list of singular vectors
    """
    intermediate = tensor
    singular_vectors, singular_values = [], []
    for mode in range(len(tensor.shape)):
        to_unfold = intermediate
        svec, sval, _ = truncated_svd(tl.unfold(to_unfold, mode), core_size[mode])
        intermediate = tl.tenalg.mode_dot(intermediate, svec.t(), mode)
        singular_vectors.append(svec)
        singular_values.append(sval)
    return intermediate, singular_vectors, singular_values


def compute_min_eigenvalue(matrix):
    # S, Ulist = hosvd(matrix)
    intermediate, evectors, evalues = sthosvd(matrix, matrix.shape[::-1])

    return [e.min().cpu().detach().numpy() for e in evalues]


