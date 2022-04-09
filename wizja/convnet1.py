from time import sleep

import torch
from torch import nn, optim, tensor, Tensor
import torch.nn.functional as F

import matplotlib.pyplot as plt

# from helpers import format_t
# from data_gen_1 import *
# from torch_helpers import *
from podstawy.torch_helpers import shuffle_samples_and_outputs
from wizja.generation.g1 import generate_sample


class ConvNet(nn.Module):
    """
        Simple NN: input(N_IN) ---> flat(hid) ---> output (N_OUT)
    """

    def __init__(self, res, n_out):
        super().__init__()
        self.res = res
        self.n_out = n_out
        self.ch1 = 5
        self.hid = 10

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)  # pozostawia RES x RES
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # RES → RES/2 (64 → 32)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5, padding=2)  # pozostawia RES x RES
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # RES → RES/4 (32 → 16)

        flat_size = 9 * (res // 4) ** 2  # 9 kanałów, RES/4 x RES/4
        self.flat_input_size = flat_size

        self.flat_in_h = nn.Linear(flat_size, self.hid, True)
        self.flat_h_h = nn.Linear(self.hid, self.hid, True)
        self.flat_h_out = nn.Linear(self.hid, n_out, True)

    def forward(self, x):
        x = x.view(-1, 3, self.res, self.res)
        y = x.copy()
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, self.flat_input_size)  # wypłaszczenie sygnału

        z = torch.cat((x, y), 0)

        x = F.relu(self.flat_in_h(x))
        x = F.relu(self.flat_h_h(x))
        x = F.relu(self.flat_h_out(x))
        return x

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


# TYP DANYCH I CPU/GPU
dtype = torch.double
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'

# GEOMETRIA SIECI
RES = 128
N_OUT = 2
N_SAMPLES = 30  # liczba próbek treningowych z każdego typu

# PROCES UCZENIA SIECI
EPOCHS = 500
REGENERATE_SAMPLES_EPOCHS = 150  # co tyle epok generujemy próbki treningowe na nowo
RESHUFFLE_EPOCHS = 45
BATCH_SIZE = 1000
LR = 0.02
MOMENTUM = 0.9

# Net creation
net = ConvNet(RES, n_out=N_OUT)
net = net.double()

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU


# fixme: UWAGA!! Przy zmianie rozmiarów sieci nie można wczytywać stanu poprzedniej ↓↓.
# net.load('saved_net_state.dat')


def generate_sample_tensors() -> tuple[Tensor, Tensor]:
    # ↓↓ to są listy pythona
    samples_tcells = generate_sample(N_SAMPLES, 'generation/tcells', RES)
    n_tcells = len(samples_tcells)
    outputs_tcells = [(1, 0) for _ in range(n_tcells)]
    n_bacter = n_tcells
    samples_bacter = generate_sample(n_bacter, 'generation/bacteria', RES)
    outputs_bacter = [(0, 1) for _ in range(n_bacter)]

    sample = torch.cat((samples_tcells, samples_bacter), 0)
    output = torch.cat((torch.tensor(outputs_tcells, dtype=dtype, device=device),
                        torch.tensor(outputs_bacter, dtype=dtype, device=device)), 0)
    if dtype == torch.double:
        sample = sample.double()
    if device == 'cuda':
        sample = sample.cuda()

    # zamiana próbek na torch.tensor (możliwa kopia do pamięci GPU)
    # t_sample_ = tensor(sample, dtype=dtype, device=device)
    # t_output_ = tensor(output, dtype=dtype, device=device)

    return sample, output


def split_to_batches(samples: Tensor, outputs: Tensor) -> tuple[list, list]:
    return torch.split(samples, BATCH_SIZE), torch.split(outputs, BATCH_SIZE)


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

    # print(f'epoch={epoch}')
    # Dodatkowe operacje w trakcie procesu uczenia
    if epoch % 10 == 9:
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
net.save('saved_net_state.dat')
print('net saved')
plt.scatter(epo_, err_)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()
