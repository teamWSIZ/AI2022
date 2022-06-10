from math import sin, tanh
from torch import nn, optim, tensor, Tensor
import torch.nn.functional as funct
import torch


def model_sinus(x):
    return 1 + sin(x)


def model_lorentz(x):
    return 1 / (1 + (x - 5) ** 2)


def model_tanh(x):
    return 1 + tanh(x * 0.2)


class ModelMultiPeak:
    peaks = []

    """
    Initialize this class with locations of lorentzian peaks (each peak[0] is scale, peak[1] is location), 
    eg. m = ModelMultiPeak([(1,0),(1,5)])    # has similar peaks at x=0 and x=5, 
    then call it via m(15)
    """

    def __init__(self, peaks) -> None:
        self.peaks = peaks

    def __call__(self, *args, **kwargs):
        x = args[0]
        val = 0
        for peak in self.peaks:
            val += peak[0] / (1 + (x - peak[1])**2)
        return val




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
    x = -20
    dx = 0.1
    vals = []
    xx = []
    peaks = []
    for i in range(10):
        peaks.extend([(0.5, i*40), (1,i*40+5)])
    model = ModelMultiPeak(peaks)    # "cardiogram"
    while x < 200:
        vals.append(model(x))
        xx.append(x)
        x += dx

    import matplotlib.pyplot as plt

    plt.plot(xx, vals, linestyle='solid')

    plt.show()
