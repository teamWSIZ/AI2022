import torch
from torch import tensor, Tensor

from convnet_model import ConvNet
from wizja.samples.generator_of_samples import generate_for_check

# TYP DANYCH I CPU/GPU
dtype = torch.double
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# GEOMETRIA SIECI
RES = 64
N_OUT = 2

# Net creation
net = ConvNet(RES, n_out=N_OUT)
net = net.double()

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU

# fixme: UWAGA!! Przy zmianie rozmiarów sieci nie można wczytywać stanu poprzedniej ↓↓.
net.load('saved_net_state.dat')


def generate_sample_tensors(sample_dir: str, n_classes=2) -> tuple[Tensor, Tensor]:
    samples_, outputs_ = generate_for_check(sample_dir, RES, n_classes)
    if dtype == torch.double:
        samples_ = samples_.double()
        outputs_ = outputs_.double()
    if device == 'cuda':
        samples_ = samples_.cuda()
        outputs_ = outputs_.cuda()
    return samples_, outputs_


SAMPLE_DIR = 'samples/leaves_test'
EXPECTED_POSITION = 1
ALLOWED_ERROR = 0.30

samples, outputs = generate_sample_tensors(SAMPLE_DIR, N_OUT)

prediction = net(samples)
results = prediction.detach().cpu().numpy()  # [ [0,1], [0,1], ...]
outputs = outputs.detach().cpu().numpy()

correct = 0
for x, e in zip(results, outputs):
    ok = max(abs(x[pos] - e[pos]) for pos in range(len(x))) < ALLOWED_ERROR
    correct += 1 if ok else 0
    print(x, e, '\t', '✔' if ok else 'x')

print(f'Poprawnie rozpoznanych próbek: {correct / len(results) * 100 :.0f}%')
