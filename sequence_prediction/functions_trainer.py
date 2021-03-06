import torch
from torch import nn, optim, tensor, Tensor

from podstawy.helpers import format_list
from sequence_prediction.functions_model import SequenceNet, model_sinus, model_tanh, model_lorentz
from sequence_prediction.sample_generator import gen_samples

dtype = torch.double
device = 'cpu'  # gdzie wykonywać obliczenia
# device = 'cuda'

HISTORY_N = 40  # ile liczb wchodzi (długość listy -- historii na podstawie której przewidujemy)
HID = 10  # ile neuronów w warstwie ukrytej

N_SAMPLE = 10000
BATCH_SIZE = 2500
EPOCHS = 2000
LR = 0.01

# Dane do uczenia sieci
DX = 0.01


def generate_sample_tensors(x_from, x_to, dx, model_function, history_len, n_samples) -> tuple[Tensor, Tensor]:
    sample, output = gen_samples(x_from, x_to, dx, model_function, history_len, n_samples)

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


def train(x_from, x_to, dx, model_function, history_len, hidden_neurons, load_net: bool, save_filename: str,
          learning_rate: float, device: str):
    # Create net, or load from a saved checkpoint
    net = SequenceNet(history_len, hidden_neurons)
    net = net.double()
    if load_net:
        net.load(save_filename)
    if device == 'cuda':
        net = net.cuda()

    # Training setup
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # będzie na GPU, jeśli gpu=True

    samples, outputs = generate_sample_tensors(x_from, x_to, dx, model_function, HISTORY_N, N_SAMPLE)
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

            # if EPOCHS - epoch < 2:
            #     # pokazujemy wyniki dla ostatnich
            #     print('---------')
            #     print(f'input: {batch_s.tolist()}')
            #     print(f'pred:{format_list(prediction.tolist())}')
            #     print(f'outp:{format_list(batch_o.tolist())}')

            total_loss += loss

            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(f' epoch:{epoch}, loss:{total_loss:.6f}')
    # Optional result save
    net.save(save_filename)
    print('net saved')


def predict(x_from, x_to, dx, model_function, history_len, hidden_neurons, saved_filename):
    net = SequenceNet(history_len, hidden_neurons)
    net = net.double()
    if saved_filename != '':
        net.load(saved_filename)

    history = [model_function(x_from + i * dx) for i in range(HISTORY_N)]  # początkowa historia

    model_values = history.copy()
    predi_values = history.copy()

    x = x_from + HISTORY_N * dx
    while x < x_to:
        history_t = tensor([history], dtype=dtype, device=device)
        history_batch = torch.split(history_t, BATCH_SIZE)
        # print(history_batch)
        nxt = net(history_batch[0])  # szukamy predykcji następnej wartości
        val = float(nxt[0][0])

        predi_values.append(val)
        model_values.append(model_function(x))

        history.append(val)
        history = history[1:]
        x += dx

    import matplotlib.pyplot as plt
    # plt.plot(history, linestyle='solid')
    plt.plot(predi_values, linestyle='dotted')
    plt.plot(model_values, linestyle='solid')

    plt.show()


if __name__ == '__main__':
    # train(x_from=2, x_to=8, dx=DX, model_function=model_lorentz, history_len=HISTORY_N, hidden_neurons=HID,
    #       load_net=True, save_filename='save.dat', learning_rate=LR, device=device)

    predict(x_from=10, x_to=11, dx=DX, model_function=model_sinus, history_len=HISTORY_N, hidden_neurons=HID,
            saved_filename='save.dat')
