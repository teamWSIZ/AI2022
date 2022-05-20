import copy
from time import sleep

import matplotlib.pyplot as plt
import torch
import torchvision.models
from torch import nn, optim, tensor, Tensor

from podstawy.torch_helpers import shuffle_samples_and_outputs
from wizja.samples.generator_of_samples import generate_sample
from convnet_model import ConvNet

# TYP DANYCH I CPU/GPU
dtype = torch.float
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# GEOMETRIA SIECI wg.
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#inception-v3
#
RES = 78
N_OUT = 4  # == num_classes
N_SAMPLES = 150  # liczba próbek treningowych z każdego typu

# PROCES UCZENIA SIECI
EPOCHS = 15000-1
REGENERATE_SAMPLES_EPOCHS = 200  # co tyle epok generujemy próbki treningowe na nowo
RESHUFFLE_EPOCHS = 10
BATCH_SIZE = 50
LR = 0.0002
MOMENTUM = 0.9

# Net creation
net = torchvision.models.inception_v3(pretrained=False, aux_logits=False)  # fixme: ←aux_logits=F jeśli chcemy RES<299
net = net.float()

# print(net)
"""
Ostatnie oryginalne warstwy Inception_v3 to: 
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
  ↑↑ ostatnią trzeba dostosować, jak mamy 3 lub 4 klasy obrazków, a nie 1000 jak w ImageNet

https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#inception-v3
"""
# fixme linia "print" ↑↑ jest potrzebna by odczytać te dwie stałe (zależne od rozdzielczości) ↓↓
# net.AuxLogits.fc = nn.Linear(768, N_OUT)  #fixme: ←zakomentowane jeśli chcemy mieć RES<299
net.fc = nn.Linear(2048, N_OUT)

# fixme: UWAGA!! Przy zmianie rozmiarów sieci nie można wczytywać stanu poprzedniej ↓↓.
net = torch.load('saved_inception_v3_eyes.dat')

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU


def generate_sample_tensors() -> tuple[Tensor, Tensor]:
    samples, outputs = generate_sample(N_SAMPLES, 'samples/eyes', RES, n_classes=N_OUT)
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
print('samples dimensions:', t_sample.shape)
print('outputs dimensions:', t_output.shape)

for epoch in range(EPOCHS):
    total_loss = tensor(0., device=device)
    n_batches = 0
    for (batch_s, batch_o) in zip(b_sample, b_output):
        optimizer.zero_grad()
        evalued = net(batch_s)
        # prediction, aux_prediction = evalued
        loss = loss_function(evalued, batch_o)

        total_loss += loss
        n_batches += 1

        loss.backward()
        optimizer.step()
        # print(loss.item())

    # print(f'epoch={epoch}')
    # Dodatkowe operacje w trakcie procesu uczenia
    if epoch % 3 == 1:
        mean_error = total_loss.item() / n_batches
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
torch.save(net, 'saved_inception_v3_eyes.dat')
print('net saved')

plt.scatter(epo_, err_)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()
