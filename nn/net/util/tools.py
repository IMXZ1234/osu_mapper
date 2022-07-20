import torch


def smooth_one_hot(prob, lambda_=1., dim=1):
    return torch.softmax(
        torch.log(prob) / lambda_,
        dim=dim
    )


def to_hard(y_soft, dim=1):
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft



if __name__ == '__main__':
    a = torch.tensor([[[-0.2847, -0.2847,  0.1197]]])
    a = torch.softmax(a, dim=-1)
    print(smooth_one_hot(a, lambda_=0.1, dim=-1))
    # print(smooth_one_hot(a, lambda_=100))
    # print(smooth_one_hot(a, lambda_=0.01))
    # b = smooth_one_hot(a, lambda_=0.01)
    # print(to_hard(b, dim=1))
