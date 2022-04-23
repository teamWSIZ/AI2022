import torch
from torch import tensor, Tensor

from convnet_model import ConvNet
from wizja.samples.g1 import generate_for_check

# TYP DANYCH I CPU/GPU
dtype = torch.double
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# GEOMETRIA SIECI
RES = 128
N_OUT = 2

# Net creation
net = ConvNet(RES, n_out=N_OUT)
net = net.double()

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# fixme: UWAGA!! Przy zmianie rozmiarów sieci nie można wczytywać stanu poprzedniej ↓↓.
net.load('saved_net_state.dat')


def generate_sample_tensors() -> tuple[Tensor, Tensor]:
    # ↓↓ to są listy pythona
    samples = generate_for_check('samples/oak_test', RES)
    if dtype == torch.double:
        samples = samples.double()
    if device == 'cuda':
        samples = samples.cuda()
    return samples


# Dane do wykresu ↓
epo_ = []
err_ = []

t_samples = generate_sample_tensors()

total_loss = tensor(0., device=device)
prediction = net(t_samples)
results = prediction.detach().cpu().numpy()  # [ [0,1], [0,1], ...]

ALLOWED_ERROR = 0.05
EXPECTED_POSITION = 0

correct = 0
print(results)
for x in results:
    correct += 1 if abs(x[EXPECTED_POSITION] - 1) < ALLOWED_ERROR else 0

print(f'Poprawnie rozpoznanych próbek: {correct / len(results) * 100 :.0f}%')
