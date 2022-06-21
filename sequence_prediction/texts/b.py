from torch import nn
import torch

enc = nn.TransformerEncoderLayer(d_model=16, nhead=8)
src = torch.rand(1, 1, 16)  # 1 batches, sequence of length 2
print(src)
out = enc(src)
print(out.size())  # (2,1,8)
print(out[0][0])
print(out[0][0].detach().numpy())
