import torch


def shuffle_samples_and_outputs(sample_, output_):
    """
        Funkcja "tasujaca" karty-próbki stosowane do uczenia sieci
    """
    size = sample_.size()[0]
    per_torch = torch.randperm(size)
    shuffled_sample = sample_[per_torch]
    shuffled_output = output_[per_torch]
    return shuffled_sample, shuffled_output
