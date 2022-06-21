from torch import nn
import torch

emb = nn.Embedding(num_embeddings=6, embedding_dim=3)

inp = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 1]])

res = emb(inp)
print(res)