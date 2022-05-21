from math import sin
from traceback import print_list

import torch
from torch import nn, optim, tensor, Tensor
import torch.nn.functional as funct

from podstawy.helpers import format_list
from sequence_prediction.functions_model import model_lorentz, SequenceNet, model_sinus
from sequence_prediction.sample_generator import gen_samples

dtype = torch.double
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'

HISTORY_N = 30  # ile liczb wchodzi (długość listy -- historii na podstawie której przewidujemy)
HID = 3  # ile neuronów w warstwie ukrytej

# liczba próbek treningowych zwracających "1"
N_SAMPLE = 10000  # liczba próbej treningowych zwracających "0"
BATCH_SIZE = 2500  # liczba próbek losowych
EPOCHS = 1500
LR = 0.1

# Czy obliczenia mają być na GPU

# Dane do uczenia sieci
DX = 0.01


def generate_sample_tensors(model_function, n_samples, history_len, x_from, x_to, dx) -> tuple[Tensor, Tensor]:
    sample, output = gen_samples(n_samples, history_len, model_function, x_from, x_to, dx)

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


def train(history_len, hidden_neurons, load_filename='', save_filename='save.dat', learning_rate=0.1, device='cpu',
          function_to_teach=model_sinus):
    # Create net, or load from a saved checkpoint
    net = SequenceNet(history_len, hidden_neurons)
    net = net.double()
    if load_filename != '':
        net.load(load_filename)
    if device == 'cuda':
        net = net.cuda()  # cała sieć kopiowana na GPU

    # Training setup
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # będzie na GPU, jeśli gpu=True

    samples, outputs = generate_sample_tensors(function_to_teach, N_SAMPLE, HISTORY_N, 0, 2, 0.1)
    b_sample, b_output = split_to_batches(samples, outputs, BATCH_SIZE)

    # Training
    for epoch in range(EPOCHS):
        total_loss = 0
        for (batch_s, batch_o) in zip(b_sample, b_output):
            optimizer.zero_grad()
            # print(batch_in)
            prediction = net(batch_s)
            prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
            loss = loss_function(prediction, batch_o)

            if EPOCHS - epoch < 2:
                # pokazujemy wyniki dla 30 ostatnich przypadków, by sprawdzić co sieć przewiduje tak naprawdę
                print('---------')
                print(f'input: {batch_s.tolist()}')
                print(f'pred:{format_list(prediction.tolist())}')
                print(f'outp:{format_list(batch_o.tolist())}')

            total_loss += loss

            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
    # Optional result save
    net.save(save_filename)
    print('net saved')


def predict(max_x, dx, history_len, hidden_neurons, saved_filename, model_function):
    net = SequenceNet(history_len, hidden_neurons)
    net = net.double()
    if saved_filename != '':
        net.load(saved_filename)

    history = [model_function(2 + i * dx) for i in range(HISTORY_N)]  # początkowa historia
    full = history.copy()
    x = HISTORY_N * dx
    while x < max_x:
        history_t = tensor([history], dtype=dtype, device=device)
        history_batch = torch.split(history_t, BATCH_SIZE)
        # print(history_batch)
        nxt = net(history_batch[0])  # szukamy predykcji następnej wartości
        val = float(nxt[0][0])
        print(f'{history} → {val}')
        full.append(val)
        history.append(val)
        history = history[1:]
        x += dx

    import matplotlib.pyplot as plt
    # plt.plot(history, linestyle='solid')
    plt.plot(full, linestyle='dotted')
    plt.show()


if __name__ == '__main__':
    # train(HISTORY_N, HID, 'save.dat', 'save.dat', LR, device, function_to_teach=model_sinus)
    predict(max_x=4, dx=DX, history_len=HISTORY_N, hidden_neurons=HID, saved_filename='save.dat',
            model_function=model_sinus)
