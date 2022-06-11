from torch import nn
import torch

rnn = nn.LSTM(input_size=3, hidden_size=3, num_layers=7)  # default(num_layers) = 1

# size for initial cell state is (num_layers, 1, input_size)
c = torch.zeros((7, 1, 3), dtype=torch.float)
h = torch.zeros((7, 1, 3), dtype=torch.float)

# whole input sequence is (n_history, 1, input_size)
inp = torch.zeros((5, 1, 3), dtype=torch.float)
o, (c, h) = rnn(inp, (c, h))

# --------------------------------
print('o ', o.size())  # [5, 1, 3]
print('c and h: ', h.size())  # [7, 1, 3]  (num_layers, hidden_size, input_size)

input_sequence = [torch.randn(1, 3) for _ in range(5)]  # list of 5 elements, each of size (1,3)
# torch.randn(1,3) ~ torch.zeros(1,3)

print(input_sequence[0])
print(input_sequence[0].view(1, 1, -1))

ff = torch.tensor([[1]])
print(ff.size()[0])