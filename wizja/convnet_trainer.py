from time import sleep

import matplotlib.pyplot as plt
import torch
from torch import nn, optim, tensor, Tensor

from podstawy.torch_helpers import shuffle_samples_and_outputs
from wizja.samples.generator_of_samples import generate_sample
from convnet_model import ConvNet

# TYP DANYCH I CPU/GPU
dtype = torch.double
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# GEOMETRIA SIECI
RES = 64
N_OUT = 3
N_SAMPLES = 30  # liczba próbek treningowych z każdego typu

# PROCES UCZENIA SIECI
EPOCHS = 20000
REGENERATE_SAMPLES_EPOCHS = 190  # co tyle epok generujemy próbki treningowe na nowo
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
net.load('saved_net_state.dat')


def generate_sample_tensors() -> tuple[Tensor, Tensor]:
    samples, outputs = generate_sample(N_SAMPLES, 'samples/cars', RES, n_classes=N_OUT)
    if dtype == torch.double:
        samples = samples.double()
        outputs = outputs.double()
    if device == 'cuda':
        samples = samples.cuda()
        outputs = outputs.cuda()
    return samples, outputs


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
print(t_sample.shape)
print(t_output.shape)

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
