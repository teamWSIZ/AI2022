from random import random

from sequence_prediction.functions_model import model_sinus


def gen_samples(n_samples, history_len, model_function, x_from, x_to, dx) -> tuple[list, list]:
    """
    Tworzymy próbki składające się z historii jakiejś funkcji (tu sin(x)), oraz z kolejnego punktu tej funkcji.
    Zadanie sieci neuronowej to przewidzieć kolejną wartość tej funkcji
    """
    samples = []
    outputs = []

    for st in range(n_samples):
        start = x_from + random() * (x_to - x_from)  # punkt startowy losowany z przedziału [xfrom,xto]
        x = [model_function(start + i * dx) for i in range(history_len)]
        samples.append(x)
        outputs.append(model_function(start + history_len * dx))
    return samples, outputs


if __name__ == '__main__':
    print(gen_samples(10, 3, model_sinus, 0, 0.2, 0.01))