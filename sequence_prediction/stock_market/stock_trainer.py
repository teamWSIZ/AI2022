from datetime import datetime

import torch
from torch import nn, optim, tensor, Tensor

from podstawy.helpers import format_list
from stock_model import *
from yfinance_sample_gen import *

dtype = torch.double
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'

HISTORY_N = 100  # ile liczb wchodzi (długość listy -- historii na podstawie której przewidujemy)
HID = 10  # ile neuronów w warstwie ukrytej

N_SAMPLE = 20000  # liczba próbek treningowych
BATCH_SIZE = 2500  # liczba próbek losowych
EPOCHS = 15300
LR = 0.01

# Dane do uczenia sieci
DX = 0.01


def generate_sample_tensors(date_from: datetime, date_to: datetime,
                            history_len, n_samples, ticker) -> tuple[Tensor, Tensor]:
    sample, output = get_samples(date_from, date_to, history_len, n_samples, ticker)

    # zamiana próbek na tensory (możliwa kopia do pamięci GPU)
    t_sample = tensor(sample, dtype=dtype, device=device)
    t_output = tensor(output, dtype=dtype, device=device)

    # przetasowanie całośći
    sample_count = t_sample.size()[0]
    per_torch = torch.randperm(sample_count)
    t_sample = t_sample[per_torch]
    t_output = t_output[per_torch]

    return t_sample, t_output


def split_to_batches(samples: Tensor, outputs: Tensor, batch_size) -> tuple[list, list]:
    return torch.split(samples, batch_size), torch.split(outputs, batch_size)


def train(date_from: datetime, date_to: datetime, history_len, n_samples, ticker,
          load_net=False, save_filename='save.dat', learning_rate=0.1, device='cpu'):
    # Create net, or load from a saved checkpoint
    net = SequenceNet(history_len, HID)
    net = net.double()
    if load_net:
        net.load(save_filename)
    if device == 'cuda':
        net = net.cuda()

    # Training setup
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # będzie na GPU, jeśli gpu=True

    samples, outputs = generate_sample_tensors(date_from, date_to, history_len, n_samples, ticker)
    b_sample, b_output = split_to_batches(samples, outputs, BATCH_SIZE)

    # Training
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch_s, batch_o) in zip(b_sample, b_output):
            optimizer.zero_grad()
            prediction = net(batch_s)
            prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
            loss = loss_function(prediction, batch_o)
            total_loss += loss

            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
        if epoch % 300 == 300-1:
            print('regenerating samples')
            samples, outputs = generate_sample_tensors(date_from, date_to, history_len, n_samples, ticker)
            b_sample, b_output = split_to_batches(samples, outputs, BATCH_SIZE)

    # Optional result save
    net.save(save_filename)
    print('net saved')


def predict(date_from: datetime, date_to: datetime, history_len, horizon, n_samples, ticker, saved_filename):
    net = SequenceNet(history_len, HID)
    net = net.double()
    net.load(saved_filename)

    histories, next_values = get_samples(date_from, date_to, history_len + horizon, n_samples, ticker)

    history = histories[0][:history_len]

    model_values = histories[0]
    predi_values = history.copy()

    for _ in range(horizon):
        history_t = tensor([history], dtype=dtype, device=device)
        history_batch = torch.split(history_t, BATCH_SIZE)
        # print(history_batch)
        nxt = net(history_batch[0])  # szukamy predykcji następnej wartości
        val = float(nxt[0][0])

        predi_values.append(val)

        history.append(val)
        history = history[1:]

    import matplotlib.pyplot as plt
    # plt.plot(history, linestyle='solid')
    plt.plot(predi_values, linestyle='dotted')
    plt.plot(model_values, linestyle='solid')

    plt.show()


if __name__ == '__main__':
    train_start = datetime(2007, 1, 1)
    train_end = datetime(2020, 1, 1)
    pred_start = datetime(2020, 1, 1)
    pred_end = datetime(2022, 5, 1)

    # train(train_start, train_end, HISTORY_N, N_SAMPLE, 'EURPLN=X', False, 'save.dat', LR, device)

    predict(pred_start, pred_end, HISTORY_N, 30, 1, 'EURPLN=X', 'save.dat')
