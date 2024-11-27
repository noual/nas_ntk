""" CNN cell for architecture search """
import torch
import torch.nn as nn
import random
from DARTS import operations as ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, random_keys):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        a=0
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                random_key = random_keys[a]
                op = ops.OPS[random_key](C, stride, affine=True)
                self.dag[i].append(op)
                a+=1


    def forward(self, s0, s1):
        # Prétraitement des entrées
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        # Initialisation des états
        states = [s0, s1]

        # Boucle sur les couches du DAG sans utiliser de poids
        for edges in self.dag:
            # Calcul du nouvel état courant sans utiliser de poids
            s_cur = sum(edge(s) for edge, s in zip(edges, states))
            states.append(s_cur)

        # Concatenation des états finaux
        s_out = torch.cat(states[2:], dim=1)

        return s_out

if __name__=="__main__":
    print(SearchCell(4, 8, 16, 32, True, True, [random.choice(list(ops.OPS.keys())) for _ in range(14)]))