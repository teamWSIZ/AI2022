import torch
from torch import nn, optim, tensor
import torch.nn.functional as funct

import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
from helpers import format_list
from data1 import *


class MyNet(nn.Module):
    """
        Simple NN: input(sz) ---> flat(hid) ---> 1
    """

    def __init__(self, sz, hid):
        super().__init__()
        self.hid = hid
        self.sz = sz
        self.flat1 = nn.Linear(sz, hid, True)
        self.flat2 = nn.Linear(hid, 1, True)

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
N = 10  # ile liczb wchodzi (długość listy)
HID = 2  # ile neuronów w warstwie ukrytej
N_SAMPLES = 1000  # liczba próbej treningowych

BATCH_SIZE = 500  # liczba próbek losowych
EPOCHS = 10000
LR = 0.0001

# Net creation
net = MyNet(N, HID)
net = net.double()

net.load('saves/n10_single_one.dat')

# Czy obliczenia mają być na GPU
if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# Próbki napewno dodatnie
# sample1, output1 = get_1()
# Próbki ujemne
# sample0, output0 = get_0()

# sample = sample1
# output = output1
# sample.extend(sample0)
# output.extend(output0)
sample, output = get_patterns(10, 0.6, n_samples=N_SAMPLES)

# zamiana próbek na tensory (możliwa kopia do pamięci GPU)
t_sample = tensor(sample, dtype=dtype, device=device)
t_output = tensor(output, dtype=dtype, device=device)
# print(t_sample)
# print(t_output)

# przetasowanie całośći
sample_count = t_sample.size()[0]
print(sample_count)
permutation = torch.randperm(sample_count)
t_sample = t_sample[permutation]
t_output = t_output[permutation]

# "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
b_sample = torch.split(t_sample, BATCH_SIZE)
b_output = torch.split(t_output, BATCH_SIZE)

# Training setup
loss_function = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # będzie na GPU, jeśli gpu=True

# Training
epo_ = []
err_ = []
for epoch in range(EPOCHS):
    total_loss = tensor(0.)
    for (batch_s, batch_o) in zip(b_sample, b_output):
        optimizer.zero_grad()
        # print(batch_in)
        prediction = net(batch_s)
        prediction = prediction.view(-1)  # size: [5,1] -> [5] (flat, same as b_out)
        loss = loss_function(prediction, batch_o)

        if EPOCHS - epoch < 30:
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
        epo_.append(epoch)
        err_.append(total_loss.item())

# Optional result save
net.save('saves/n10_single_one.dat')
print('net saved')
plt.scatter(epo_, err_)
plt.show()