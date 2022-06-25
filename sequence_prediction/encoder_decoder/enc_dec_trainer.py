from random import randint

from enc_dec_model import *
from torch import tensor, nn, functional as F, optim
import torch
import matplotlib.pyplot as plt

ALPHABET = 300  # ile mamy tokenów... czyli w przypadku tekstów: ile różnych słów dopuszczamy...
SAMPLES = 20000
ENCODED_SIZE = 4  # tyle liczb typu "float" pozostaje po zakodowaniu danego znaku

BATCH_SIZE = 1500
EPOCHS = 20
LR = 0.01

# Model
net = EncoderDecoderModule(ALPHABET, ninp=ENCODED_SIZE)


net.load('enc_dec.dat')


def get_batches():
    # ↓↓ to są listy pythona
    sample = [randint(0, ALPHABET - 1) for _ in range(SAMPLES)]
    output = sample.copy()

    # zamiana próbek na torch.tensor (możliwa kopia do pamięci GPU)
    t_sample = tensor(sample)

    # "krojenie" próbek na "batches" (grupy próbek, krok optymalizacji po przeliczeniu całej grupy)
    bs = torch.split(t_sample, BATCH_SIZE)
    t_output = tensor(output)
    bo = torch.split(t_output, BATCH_SIZE)
    return bs, bo


b_sample, b_output = get_batches()

# ####
# Training setup
loss_function = nn.MSELoss(reduction='mean')
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=LR)


def transform_to_01(batch):
    res = []
    for elem in batch:
        idx = elem.item()
        e01 = [0] * ALPHABET
        e01[idx] = 1
        res.append(e01)
    return torch.tensor(res).float()


# Training
print('starting training...')
epo_ = []
err_ = []
for epoch in range(EPOCHS):
    total_loss = tensor(0.)
    elems = 0
    for (batch_s, batch_o) in zip(b_sample, b_output):
        optimizer.zero_grad()
        # print(batch_in)
        prediction = net(batch_s)
        # print(prediction.size())  # (batch, alphabet)
        # print(batch_o.size())  # (batch)
        elems += len(batch_o)
        bf = transform_to_01(batch_o)
        # print(prediction.size())  # (batch, alphabet)
        # print(bf.size())  # (batch, alphabet)

        prediction = prediction
        loss = loss_function(prediction.view(-1), bf.view(-1))

        # if EPOCHS - epoch < 20:
        # print('---------')
        # print(f'input: {batch_s.tolist()}')
        # print(f'pred:{prediction.view(-1)[:30]}')
        # print(f'outp:{bf.view(-1)[:30]}')

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
        # t_sample, t_output = shuffle_samples_and_outputs(t_sample, t_output)
        # b_sample = torch.split(t_sample, BATCH_SIZE)
        # b_output = torch.split(t_output, BATCH_SIZE)

print('-' * 50)
# x = [1, 4, 8, 25, 2, ]
x = [i for i in range(ALPHABET)]
xt = torch.tensor(x)
xt_ = net(xt)
good = 0
for i in range(len(x)):
    correct = torch.argmax(transform_to_01(xt)[i]).item()
    result = torch.argmax(xt_[i]).item()
    print('correct: ', correct)
    print('result : ', result)
    if correct == result: good += 1
    print('---')
print(f'in total: {100. * good / len(x) : .1f}% correct')

# Optional result save
net.save('enc_dec.dat')
print('net saved')
plt.scatter(epo_, err_)
plt.show()
