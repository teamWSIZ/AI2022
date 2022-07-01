from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import tensor, nn, optim
from corpus_generators import *

from transformer_model import TransformerModel

ALPHABET = 30  # ile mamy tokenów... czyli w przypadku tekstów: ile różnych słów dopuszczamy...
SAMPLES = 2000
LEN = 2000

ENCODED_SIZE = 30  # tyle liczb typu "float" pozostaje po zakodowaniu danego znaku
BPTT = 60  #todo: size of each sequence fed to net, "history length"

BATCH_SIZE = 100
EPOCHS = 5
LR = 0.001
# ADAM = False
ADAM = True

# device = torch.device('cuda')
device = torch.device('cpu')


###############################################################################
# Build the model
###############################################################################

def create_net(load=True):
    ntokens = ALPHABET
    net = TransformerModel(ntokens, ninp=ENCODED_SIZE, nhead=5, nhid=50, nlayers=2, dropout=0.1).to(device)
    if load: net.load('transf_save.dat')
    return net




def get_batches(n_samples) -> tuple[tensor, tensor]:
    """
    :return: Tuple[Tensor]; each of size = (batch, sample, _internalrepresentation_)
    """
    # data = get_journey(LEN, max_distance=20)
    # data = get_periodic(length=LEN, alphabet=ALPHABET)
    # data = get_small_samples(length=LEN)
    data = get_periodic_samples(length=LEN, dist=10)

    # print('example data:', data[:BPTT * 2])
    starts = [randint(0, LEN - BPTT - 1) for _ in range(n_samples)]

    samples = []
    outputs = []
    for s in starts:
        samples.append(data[s:s + BPTT])
        outputs.append(data[s + 1:s + 1 + BPTT])
    t_sample = tensor(samples).to(device)  # (batch, sample, internal)
    t_output = tensor(outputs).to(device)
    bs = torch.split(t_sample, BATCH_SIZE)
    bo = torch.split(t_output, BATCH_SIZE)
    return bs, bo


def transform_to_01(t):
    # input = (sample, batch)
    batch_idx_max = t.size()[1]

    res = torch.zeros(BPTT, batch_idx_max, ALPHABET)
    for b in range(batch_idx_max):
        for s in range(BPTT):
            res[s][b][t[s][b]] = 1.
    return res.to(device)


def train():
    net = create_net(load=LOAD)

    # Training setup
    loss_function = nn.MSELoss(reduction='mean')  # [0,1,0] vs [1,0,0] → 2/3 = 0.667
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    if ADAM: optimizer = optim.Adam(net.parameters(), lr=LR)

    # Training
    print('starting training...')
    net.train()  # dropout & gradinets are "on"

    epo_ = []
    err_ = []
    b_sample, b_output = get_batches(SAMPLES)

    for epoch in range(EPOCHS):
        total_loss = tensor(0.).to(device)
        elems = 0
        st = datetime.now().timestamp()
        for (batch_s, batch_o) in zip(b_sample, b_output):
            optimizer.zero_grad()

            # model takes (sequence, batch, _internalrepr_) tensors
            input_ = torch.transpose(batch_s, 0, 1)
            expect_ = torch.transpose(batch_o, 0, 1)

            prediction = net(input_)
            elems += expect_.size()[1]

            expect_01 = transform_to_01(expect_)
            # print(prediction.size())  # (batch, alphabet)
            # print(bf.size())  # (batch, alphabet)
            loss = loss_function(prediction.view(-1), expect_01.view(-1))

            total_loss += loss
            loss.backward()
            optimizer.step()
        # Dodatkowe operacje w trakcie procesu uczenia
        if epoch % 5 == 0:
            # print(f'total elems: {elems}')  # == SAMPLES
            # print('exloss:',
            #       loss_function(tensor([0, 1, 0], dtype=torch.float32), tensor([1, 0, 0], dtype=torch.float32)))
            en = datetime.now().timestamp()
            print(
                f' epoch:{epoch}, time/epoch: {en - st:.1f}s, loss (~ %errors):{100 * 3 / 2 * total_loss / elems:.6f}')
            # assume 1 error → 0.666
            epo_.append(epoch)
            err_.append(100*3/2*total_loss.item()/elems)

    # # Optional result save
    net.save('transf_save.dat')
    print('net saved')
    # plt.scatter(epo_, err_)
    # plt.show()


def predict(steps=100):
    net = create_net()
    net.eval()  # no dropout, no gradients

    history, _ = get_batches(1)
    history = history[0]  # tuple → 0'th tensor; (batch, sample_idx) → token
    full = torch.clone(history)
    OFF = -BPTT
    print('before:', full[0, OFF:])
    for _ in range(steps):
        input_ = torch.transpose(history, 0, 1)  # (sample, batch) → token
        nxt = net(input_)  # szukamy predykcji następnej wartości → (sample, batch) → probabilities
        # get token with the highest probability;
        tokens = torch.argmax(nxt, dim=2)  # (sample,batch) –> token
        token = tokens[-1, 0]   # last generated = generated with full knowledge of history
        # cat
        token_t = tensor([[token]], dtype=torch.int)
        history = torch.cat((history, token_t), dim=1)
        full = torch.cat((full, token_t), dim=1)
        history = history[:, 1:]
    print('after: ', full[0, OFF - steps:])


if __name__ == '__main__':
    # LOAD = False
    LOAD = True
    for i in range(1): train()
    predict()
