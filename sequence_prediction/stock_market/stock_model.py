from math import sin, tanh
from torch import nn, optim, tensor, Tensor
import torch.nn.functional as funct
import torch
from yfinance_sample_gen import *


class SequenceNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, input_size, hid):
        super().__init__()
        self.hid = hid
        self.sz = input_size
        self.flat1 = nn.Linear(input_size, hid, True)
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
    x = -10
    dx = 0.1
    vals, outp = get_samples(datetime(2021, 2, 1), datetime.now(),80, 1, 'AAPL')

    import matplotlib.pyplot as plt

    plt.plot(vals[0], linestyle='solid')

    plt.show()
