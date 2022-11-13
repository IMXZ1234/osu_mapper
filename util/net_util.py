import torch


def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, (list, tuple)):
        for net_ in net:
            grad_clipping(net_, theta)
    else:
        params = [p for p in net.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm
