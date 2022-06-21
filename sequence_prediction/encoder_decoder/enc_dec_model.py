import torch
import torch.nn.functional as F
from torch import nn, Tensor, zeros


class EncoderDecoderModule(nn.Module):
    """
    Encodes sequences of integers to few-float representation, and tries to win them back.

    """

    def __init__(self, ntoken, ninp):
        super(EncoderDecoderModule, self).__init__()
        self.encoder = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)  # input =[128], albo input = [314]
        # - - -
        self.decoder = nn.Linear(ninp, ntoken)  # output = [0.13,0.11,0.001,0.75,...] (ntoken liczb)

        self.ntoken = ntoken

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x: Tensor):
        mid = self.encoder(x)
        output = F.softmax(self.decoder(mid), dim=-1)
        return output

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


if __name__ == '__main__':
    enc = nn.Embedding(10, 5)
    dec = nn.Linear(5, 7)

    inp = torch.tensor([1, 2, 2, 9])
    print(inp)
    middle = enc(inp)
    print(middle)
    out = F.log_softmax(dec(middle), dim=-1)
    print(out)
    print(out.view(-1, 10))
