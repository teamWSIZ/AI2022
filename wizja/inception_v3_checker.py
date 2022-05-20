import torch
import torchvision
from torch import nn, Tensor

from wizja.samples.generator_of_samples import generate_for_check

# TYP DANYCH I CPU/GPU
dtype = torch.float
# device = 'cpu'  # gdzie wykonywać obliczenia
device = 'cuda'

# GEOMETRIA SIECI
RES = 78
N_OUT = 4

# Net creation
net = torchvision.models.inception_v3(pretrained=False, aux_logits=False)  # fixme: ←aux_logits=F jeśli chcemy RES<299
net = net.float()
# net.AuxLogits.fc = nn.Linear(768, N_OUT)# fixme: ←zakomentowane jeśli chcemy RES<299
net.fc = nn.Linear(2048, N_OUT)
net = torch.load('saved_inception_v3_eyes.dat')  # fixme: uwaga czy geometria modelu zgadza się z geometrią save'a
net.training = False

if device == 'cuda':
    net = net.cuda()  # cała sieć kopiowana na GPU


def generate_sample_tensors(sample_dir: str, n_classes=2) -> tuple[Tensor, Tensor]:
    samples_, outputs_ = generate_for_check(sample_dir, RES, n_classes)
    if dtype == torch.double:
        samples_ = samples_.double()
        outputs_ = outputs_.double()
    if device == 'cuda':
        samples_ = samples_.cuda()
        outputs_ = outputs_.cuda()
    return samples_, outputs_


SAMPLE_DIR = 'samples/eyes_test'
ALLOWED_ERROR = 0.30

samples, outputs = generate_sample_tensors(SAMPLE_DIR, N_OUT)

evalued = net(samples)
# prediction, aux_prediction = evalued
prediction = evalued
# probabilities = torch.nn.functional.softmax(prediction, dim=1)
# print(prediction[0])
# print(probabilities[0])

results = prediction.detach().cpu().numpy()  # [ [0,1], [0,1], ...]
outputs = outputs.detach().cpu().numpy()

correct = 0
for x, e in zip(results, outputs):
    ok = max(abs(x[pos] - e[pos]) for pos in range(len(x))) < ALLOWED_ERROR
    correct += 1 if ok else 0
    print(x, e, '\t', '✔' if ok else 'x')

print(f'Poprawnie rozpoznanych próbek: {correct / len(results) * 100 :.0f}%')
