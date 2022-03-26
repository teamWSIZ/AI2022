from time import sleep

from torch import nn, optim, tensor, Tensor
import torch.nn.functional as funct

import matplotlib.pyplot as plt

from helpers import format_t
from data_gen_1 import *
from torch_helpers import *


class MyNet(nn.Module):
    def __init__(self, n_in, hid, n_out, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.hid = hid
        self.n_in = n_in

        # alternative: deep network: 4 hidden layers of 5 neurons each
        self.num_hidden_layers = 3  #todo: ważne parametry sieci!!!!
        self.size_hidden_layer = 5
        self.flat_in_h = nn.Linear(n_in, self.size_hidden_layer, True)
        # ↓↓ nie może być zwykła lista pythona, bo źle wtedy działa konwersja parametrów i zapis
        self.hh = nn.ModuleList(
            [nn.Linear(self.size_hidden_layer, self.size_hidden_layer, True) for _ in range(self.num_hidden_layers)])
        self.flat_h_out = nn.Linear(self.size_hidden_layer, n_out, True)

    def forward(self, x):
        """ Main function for evaluation of input """
        if self.dropout_rate > 0:
            # niektóre pozycje próbek będą losowo zerowane
            x = nn.Dropout(self.dropout_rate)(x)
        x = x.view(-1, self.n_in)

        x = funct.relu(self.flat_in_h(x))
        for i in range(self.num_hidden_layers):
            x = funct.relu(self.hh[i](x))
        x = funct.relu(self.flat_h_out(x))
        return x

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


dtype = torch.double
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'
N_IN = 10  # ile liczb wchodzi (długość listy)
HID = 128  # ile neuronów w warstwie ukrytej
# MASKS = ['111', '000', '101', '11011']
MASKS = ['111', '000']
N_OUT = len(MASKS) + 1  # ostatnia to przypadek, gdy żadnej maski nie wykryto
N_SAMPLES = 10000  # liczba próbek treningowych

EPOCHS = 2000
REGENERATE_SAMPLES_EPOCHS = 800  # co tyle epok generujemy próbki treningowe na nowo
RESHUFFLE_EPOCHS = 100
BATCH_SIZE = 500
LR = 0.005  # learning rate -- jak dużo zmieniać wagi sieci co "BATCH_SIZE" próbek
MOMENTUM = 0.9
LOAD = False
# LOAD = True

# Net creation
net = MyNet(N_IN, HID, N_OUT, dropout_rate=0.00)
net = net.double()

# fixme: UWAGA!! Przy zmianie rozmiarów sieci nie można wczytywać stanu poprzedniej ↓↓.
if LOAD:
    net.load('saves/n10_single_one.dat')

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU


def generate_sample_tensors() -> tuple[Tensor, Tensor]:
    # ↓↓ to są listy pythona
    sample, output = get_patterns_multimask(N_IN, probability=0.3, n_samples=N_SAMPLES, masks=MASKS)

    # zamiana próbek na torch.tensor (możliwa kopia do pamięci GPU)
    t_sample_ = tensor(sample, dtype=dtype, device=device)
    t_output_ = tensor(output, dtype=dtype, device=device)

    return t_sample_, t_output_


def split_to_batches(samples: Tensor, outputs: Tensor) -> tuple[list, list]:
    return torch.split(samples, BATCH_SIZE), torch.split(outputs, BATCH_SIZE)


def print_few_predictions(inputs: Tensor, prediction: Tensor, outputs: Tensor, n):
    print(f'input: {format_t(inputs, n_few=n)}')
    print(f'pred:{format_t(prediction, n_few=n)}')
    print(f'outp:{format_t(outputs, n_few=n)}')


# ####
# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)  # będzie na GPU, jeśli gpu=True

# Training
print('starting training...')
sleep(0.5)
# Dane do wykresu ↓
epo_ = []
err_ = []

t_sample, t_output = generate_sample_tensors()
b_sample, b_output = split_to_batches(t_sample, t_output)

for epoch in range(EPOCHS):
    total_loss = tensor(0., device=device)
    for (batch_s, batch_o) in zip(b_sample, b_output):
        optimizer.zero_grad()
        prediction = net(batch_s)
        loss = loss_function(prediction, batch_o)

        total_loss += loss
        loss.backward()
        optimizer.step()

        if EPOCHS - epoch < 5:
            print_few_predictions(batch_s, prediction, batch_o, n=5)

    # Dodatkowe operacje w trakcie procesu uczenia
    if epoch % 20 == 19:
        mean_error = total_loss.item() / len(b_sample)
        print(f' epoch:{epoch}, loss:{mean_error:.6f}')
        epo_.append(epoch)
        err_.append(mean_error)

    if epoch % RESHUFFLE_EPOCHS == 0:
        print('shuffle')
        t_sample, t_output = shuffle_samples_and_outputs(t_sample, t_output)
        b_sample, b_output = split_to_batches(t_sample, t_output)

    if epoch % REGENERATE_SAMPLES_EPOCHS == 0:
        print('generating new samples')
        t_sample, t_output = generate_sample_tensors()
        b_sample, b_output = split_to_batches(t_sample, t_output)

# Optional result save
net.save('saves/n10_single_one.dat')
print('net saved')
plt.scatter(epo_, err_)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()
