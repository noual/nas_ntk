import numpy as np
import torch


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld


class NASWOT:

    def __init__(self, batch_size, device=torch.device("cpu")):
        self.batch_size = batch_size
        self.K = np.zeros((batch_size, batch_size))
        self.device = device

    def reset(self):
        self.K = np.zeros((self.batch_size, self.batch_size))

    def score(self, network, inputs):
        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                self.K = self.K + K.cpu().numpy() + K2.cpu().numpy()
            except:
                pass

        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        for name, module in network.named_modules():
            if 'ReLU' in str(type(module)):
                # hooks[name] = module.register_forward_hook(counting_hook)
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

        network = network.to(self.device)
        x = inputs
        x = x.to(self.device)
        x2 = torch.clone(x).to(self.device)
        jacobs,  y = get_batch_jacobian(network, x)

        network(x)
        s = hooklogdet(self.K)
        return s/100


def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)[1]
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, y.detach()