from math import sin
from torch import nn, optim, tensor, Tensor
import torch.nn.functional as funct
import torch


def model_sinus(x):
    return 1 + sin(x)


def model_lorentz(x):
    return 1 / (1 + (x - 5) ** 2)


class SequenceNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, sz, hid):
        super().__init__()
        self.hid = hid
        self.sz = sz
        self.flat1 = nn.Linear(sz, hid, True)
        self.flat2 = nn.Linear(hid, 1, True)

    def forward(self, x):
        """ Main function for evaluation of input """
        x = x.view(-1, self.sz)
        # print(x.size())  # batchsize x self.sz
        x = self.flat1(x)
        # print(x.size())  # batchsize x self.hid
        x = self.flat2(funct.relu(x))
        return funct.relu(x)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


if __name__ == '__main__':
    # todo: plot these functions
    pass
