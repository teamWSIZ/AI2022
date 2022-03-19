from time import sleep

import torch
from torch import nn, optim, tensor
import torch.nn.functional as funct

import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from helpers import format_list
from data_gen_1 import *
from torch_helpers import *


class MyNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, sz, hid, n_out):
        super().__init__()
        self.hid = hid
        self.sz = sz
        self.flat1 = nn.Linear(sz, hid, True)
        self.flat2 = nn.Linear(hid, n_out, True)

    def forward(self, x):
        """ Main function for evaluation of input """
        x = x.view(-1, self.sz)
        # print(x.size())  # batchsize x self.sz
        x = self.flat1(x)
        # print(x.size()) # batchsize x self.hid
        x = self.flat2(funct.relu(x))
        return funct.relu(x)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def save(self, filename):
        torch.save(self.state_dict(), filename)


dtype = torch.double
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'
N_IN = 8  # ile liczb wchodzi (długość listy)
HID = 2  # ile neuronów w warstwie ukrytej
N_OUT = 1
N_SAMPLES = 2000  # liczba próbek treningowych
probability1 = 0.20

BATCH_SIZE = 200  # liczba próbek losowych
EPOCHS = 1000
LR = 0.01

# Net creation
net = MyNet(N_IN, HID, N_OUT)
net = net.double()

# odkomentowac jesli chcemy wczytac siec nauczona poprzednio
# net.load('saves/n10_single_one.dat')

# Czy obliczenia mają być na GPU
if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# ↓↓ to są listy pythona
sample, output = get_patterns_single_1(N_IN, probability1, n_samples=N_SAMPLES)

# zamiana próbek na torch.tensor (możliwa kopia do pamięci GPU)
t_sample = tensor(sample, dtype=dtype, device=device)
t_output = tensor(output, dtype=dtype, device=device)

# "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
b_sample = torch.split(t_sample, BATCH_SIZE)
b_output = torch.split(t_output, BATCH_SIZE)

# ####
# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

# Training
print('starting training...')
# sleep(0.5)
epo_ = []
err_ = []
for epoch in range(EPOCHS):
    total_loss = tensor(0.)
    for (batch_s, batch_o) in zip(b_sample, b_output):
        optimizer.zero_grad()
        # print(batch_in)
        prediction = net(batch_s)
        prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
        # print(prediction.size())
        # print(batch_o.size())
        loss = loss_function(prediction, batch_o.view(-1))

        if EPOCHS - epoch < 20:
            # pokazujemy wyniki dla ostatnich przypadków, by sprawdzić co sieć przewiduje tak naprawdę
            print('---------')
            print(f'input: {batch_s.tolist()}')
            print(f'pred:{format_list(prediction.view(-1).tolist())}')
            print(f'outp:{format_list(batch_o.view(-1).tolist())}')

        total_loss += loss

        loss.backward()
        optimizer.step()

    # Dodatkowe operacje w trakcie procesu uczenia
    if epoch % 20 == 0:
        print(f' epoch:{epoch}, loss:{total_loss:.6f}')
        epo_.append(epoch)
        err_.append(total_loss.item())

    if epoch % 500 == 0:
        print('shuffle')
        t_sample, t_output = shuffle_samples_and_outputs(t_sample, t_output)
        b_sample = torch.split(t_sample, BATCH_SIZE)
        b_output = torch.split(t_output, BATCH_SIZE)

# Optional result save
net.save('saves/n10_single_one.dat')
print('net saved')
plt.scatter(epo_, err_)
plt.show()
