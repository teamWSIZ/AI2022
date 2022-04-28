from time import sleep

import matplotlib.pyplot as plt
import torch
from torch import nn, optim, tensor, Tensor


from podstawy.torch_helpers import shuffle_samples_and_outputs
from wizja.samples.g1 import generate_sample
from convnet_model import ConvNet


# TYP DANYCH I CPU/GPU
dtype = torch.double
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# GEOMETRIA SIECI
RES = 128
N_OUT = 2
N_SAMPLES = 30  # liczba próbek treningowych z każdego typu

# PROCES UCZENIA SIECI
EPOCHS = 200
REGENERATE_SAMPLES_EPOCHS = 90  # co tyle epok generujemy próbki treningowe na nowo
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
    # ↓↓ to są listy pythona
    samples_type1 = generate_sample(N_SAMPLES, 'samples/carfront', RES)
    n_1 = len(samples_type1)
    outputs_type1 = [(1, 0) for _ in range(n_1)]
    n_2 = n_1
    samples_type2 = generate_sample(n_2, 'samples/carback', RES)
    n_2 = len(samples_type2)
    outputs_type2 = [(0, 1) for _ in range(n_2)]

    sample = torch.cat((samples_type1, samples_type2), 0)
    output = torch.cat((torch.tensor(outputs_type1, dtype=dtype, device=device),
                        torch.tensor(outputs_type2, dtype=dtype, device=device)), 0)
    if dtype == torch.double:
        sample = sample.double()
    if device == 'cuda':
        sample = sample.cuda()

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
