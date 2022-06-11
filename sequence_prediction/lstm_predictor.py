from torch import nn, optim, tensor, Tensor, zeros
import torch.nn.functional as funct
import torch



class LSTM_Predictor(nn.Module):
    """
    Predictor using long short-term memory cells, based on
    https://www.python-engineer.com/posts/pytorch-time-sequence/
    """

    def __init__(self, n_hidden=51, device='cpu'):
        super(LSTM_Predictor, self).__init__()
        print(f'Creating model with number of hidden params for LSTMs = {n_hidden}')
        self.n_hidden = n_hidden
        self.device = device

        # lstm1 > lstm2 > linear
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x: Tensor, future=0):
        # x : (batch_size, input_size==HISTORY_N)
        # print(x.size())

        # dont do future prediction per default
        outputs = []
        n_samples = x.size(0)

        h_t = zeros(n_samples, self.n_hidden, dtype=torch.double, device=self.device)  # hidden state for lstm1
        c_t = zeros(n_samples, self.n_hidden, dtype=torch.double, device=self.device)  # initial cell state for lstm1

        h_t2 = zeros(n_samples, self.n_hidden, dtype=torch.double, device=self.device)  # hidden state for lstm1
        c_t2 = zeros(n_samples, self.n_hidden, dtype=torch.double, device=self.device)  # initial cell state for lstm2

        output = None
        for in_t in x.split(split_size=1, dim=1):
            # cut a given input into pieces of size "1" -- show them to LSTM1 input 1 by 1
            # (batch_size, 1)
            # print(f'in_t.size={in_t.size()}')   # (2500, 1)
            h_t, c_t = self.lstm1(in_t, (h_t, c_t))  # recursion: input + (hidden,cell) → new (hidden,cell)
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
        # print(f'output.size={output.size()}')
        outputs.append(output)

        # default: unused
        # for i in range(future):
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


