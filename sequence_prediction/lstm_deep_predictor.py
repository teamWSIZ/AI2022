from torch import nn, optim, tensor, Tensor, zeros
import torch.nn.functional as funct
import torch


class LSTM_DeepPredictor(nn.Module):
    """
    Predictor using long short-term memory cells, based on
    https://www.python-engineer.com/posts/pytorch-time-sequence/
    """

    def __init__(self, n_history, n_features, n_hidden=51, n_layers=4, device='cpu'):
        super(LSTM_DeepPredictor, self).__init__()
        print(f'Creating model with number of hidden params for LSTMs = {n_hidden}')
        self.n_history = n_history
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x: Tensor, future=0):
        # x : (batch_size, HISTORY_N);
        batch_size = x.size()[0]
        outputs = []

        # initial hidden + cell state for lstm
        # size for initial cell state is (num_layers, 1, input_size)
        h_t = zeros((self.n_layers, batch_size, self.n_hidden), dtype=torch.double, device=self.device)
        c_t = zeros((self.n_layers, batch_size, self.n_hidden), dtype=torch.double, device=self.device)

        # self.lstm expects input with dimensions: (n_history, n_batches, n_features),
        # if batch_first=True →  (n_batches, n_history, n_features)
        x = x.view(-1, self.n_history, self.n_features)

        x, (h, c) = self.lstm(x, (h_t, c_t))
        output = torch.nn.functional.relu(self.linear(x))
        # print(output.size())  # (100,1,1) (?)
        outputs.append(output[:,-1])

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)
