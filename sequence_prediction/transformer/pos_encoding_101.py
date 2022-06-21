from model import *
import torch

p = PositionalEncoding(d_model=4, dropout=0, max_len=20)
"""
Shape:
    x: [sequence length, batch size, embed dim]
    output: [sequence length, batch size, embed dim]
    
    https://youtu.be/dichIcUZfOw?t=674
"""
sequences = [[0, 0, 1, 1], [0, 1, 0, 1], [0, 2, 2, 1]]  # single sequence, at each "T" represented by 4 numbers
t = torch.tensor(sequences)
t = t.unsqueeze(dim=1)
print(t.size())  # [3,1,4]
r = p.forward(t)
print(r.size())  # [3,1,4]
print(r[0][0])  # encoded 0th element of the sequence → tensor([0, 0, 1, 1])
print(r[1][0])  # encoded 1st element of the sequence → tensor([0.8415, 1.5403, 0.0100, 1.9999])
print(r[2][0])  # encoded 1st element of the sequence → tensor([0.9093, 1.5839, 2.0200, 1.9998])
