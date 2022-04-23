
import torch
import torch.nn.functional as F
from torch import nn


class ConvNet(nn.Module):
    """
        Simple NN: input(N_IN) ---> flat(hid) ---> output (N_OUT)
    """

    def __init__(self, res, n_out):
        super().__init__()
        self.res = res
        self.n_out = n_out
        self.ch1 = 5
        self.hid = 10

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)  # pozostawia RES x RES
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # RES → RES/2 (64 → 32)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5, padding=2)  # pozostawia RES x RES
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # RES → RES/4 (32 → 16)

        flat_size = 9 * (res // 4) ** 2  # 9 kanałów, RES/4 x RES/4
        self.flat_input_size = flat_size

        self.flat_in_h = nn.Linear(flat_size, self.hid, True)
        self.flat_h_h = nn.Linear(self.hid, self.hid, True)
        self.flat_h_out = nn.Linear(self.hid, n_out, True)

    def forward(self, x):
        x = x.view(-1, 3, self.res, self.res)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, self.flat_input_size)  # wypłaszczenie sygnału

        x = F.relu(self.flat_in_h(x))
        x = F.relu(self.flat_h_h(x))
        x = F.relu(self.flat_h_out(x))
        return x

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)