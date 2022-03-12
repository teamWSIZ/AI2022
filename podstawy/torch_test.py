import torch
from torch import tensor

a = tensor([0, 1, 2, 7.0])
print(torch.sum(a).item())  # tensor(10)
