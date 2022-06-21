import torch
from torch import nn, optim, tensor, Tensor

from podstawy.helpers import format_list
from sequence_prediction.functions_model import *
from sequence_prediction.lstm_deep_predictor import LSTM_DeepPredictor
from sequence_prediction.lstm_predictor import LSTM_Predictor
from sequence_prediction.sample_generator import gen_samples

dtype = torch.double
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

HISTORY_N = 400  # ile liczb wchodzi (długość listy -- historii na podstawie której przewidujemy)
HID = 15  # rozmiar wyjścia z LSTM ~~ilość pamięci LSTM'ów
N_LAYERS = 3   # liczba wasrstw LSTM'ów

N_SAMPLE = 1000
BATCH_SIZE = 250
EPOCHS = 100
LR = 0.1

# Dane do uczenia sieci
DX = 0.5  # HISTORY_N = 60 oznacza przedział HISTORY_N * DX


def generate_sample_tensors(x_from, x_to, dx, model_function, history_len, n_samples) -> tuple[Tensor, Tensor]:
    sample, output = gen_samples(x_from, x_to, dx, model_function, history_len, n_samples)

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


def train(x_from, x_to, dx, model_function, history_len, hidden_neurons, load_net: bool, save_filename: str,
          learning_rate: float, device: str):
    print(
        f'training the model on interval: [{x_from},{x_to}]; history: {history_len} → interval of length: {history_len * dx}')
    # Create net, or load from a saved checkpoint
    # net = LSTM_Predictor(hidden_neurons, device)
    net = LSTM_DeepPredictor(n_history=history_len, n_features=1, n_hidden=hidden_neurons, n_layers=N_LAYERS,
                             device=device)
    net = net.double()
    if load_net:
        net.load(save_filename)
    if device == 'cuda':
        net = net.cuda()

    # Training setup
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    samples, outputs = generate_sample_tensors(x_from, x_to, dx, model_function, HISTORY_N, N_SAMPLE)
    b_sample, b_output = split_to_batches(samples, outputs, BATCH_SIZE)

    # Training
    for epoch in range(EPOCHS):
        total_loss = 0
        net.train()

        for (batch_s, batch_o) in zip(b_sample, b_output):
            # print(batch_s.size(), batch_o.size())  # (1,100), (1)
            optimizer.zero_grad()
            prediction = net(batch_s)
            prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
            # print(prediction.size())    #(1)
            loss = loss_function(prediction, batch_o)

            total_loss += loss
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
    # Optional result save
    net.save(save_filename)
    print('net saved')


def predict(x_from, x_to, dx, history_len, model_function, hidden_neurons, saved_filename, device):
    print(f'predicting series for interval: [{x_from},{x_to}]; '
          f'history: {history_len} → interval of length: {history_len * dx}')

    # net = LSTM_Predictor(hidden_neurons, device)
    net = LSTM_DeepPredictor(n_history=history_len, n_features=1, n_hidden=hidden_neurons, n_layers=N_LAYERS,
                             device=device)

    net = net.double()
    if saved_filename != '':
        net.load(saved_filename)
    if device == 'cuda':
        net = net.cuda()

    history = [model_function(x_from + i * dx) for i in range(HISTORY_N)]  # początkowa historia
    xx = [x_from + i * dx for i in range(HISTORY_N)]

    model_values = history.copy()
    predi_values = history.copy()

    x = x_from + HISTORY_N * dx
    while x < x_to:
        history_t = tensor([history], dtype=dtype, device=device)
        history_batch = torch.split(history_t, BATCH_SIZE)
        nxt = net(history_batch[0])  # szukamy predykcji następnej wartości
        val = float(nxt[0][0])

        predi_values.append(val)
        model_values.append(model_function(x))
        xx.append(x)

        history.append(val)
        history = history[1:]
        x += dx

    import matplotlib.pyplot as plt
    # plt.plot(history, linestyle='solid')
    plt.plot(xx, predi_values, linestyle='dotted')
    plt.plot(xx, model_values, linestyle='solid')

    plt.show()


if __name__ == '__main__':
    # model = model_sinus
    peaks = []
    for i in range(15):
        peaks.extend([(0.5, i * 40), (1, i * 40 + 5)])
    model = ModelMultiPeak(peaks)  # "cardiogram"

    for i in range(5):
        print(f'iteration {i}')
        train(x_from=-20, x_to=400, dx=DX, model_function=model, history_len=HISTORY_N, hidden_neurons=HID,
              load_net=True, save_filename='save.dat', learning_rate=LR, device=device)

    predict(x_from=-5, x_to=400, dx=DX, history_len=HISTORY_N, model_function=model, hidden_neurons=HID,
            saved_filename='save.dat', device=device)
