from random import randint

import torch
from PIL import Image, ImageEnhance
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as TS
import torchvision.transforms.functional as F


def convert_index_to_bin_tensor(t: tensor, n_classes):
    """
    [1,2,0] -> [[0,1,0],[0,0,1],[1,0,0]]
    """
    w = [[1. if i.data == k else 0. for k in range(n_classes)] for i in t]
    return tensor(w)


class SuperposeSign(object):
    """
    Klasa którą można użyć jako "Transformację" (jak: torchvision.transforms) w kompozycji transformacji
    obrazów.
    """
    sign: Image
    mask: Image

    def __init__(self, sign_filename='sign.png'):
        sign = Image.open(sign_filename).resize((64, 64))
        self.sign = sign
        self.mask = sign.copy().convert('L')  # 'cień'; "single channel image"
        self.mask = ImageEnhance.Brightness(self.mask).enhance(4)  # maska jest typu ~0.4; (zbyt transparentna)
        # self.mask.show()

    def __call__(self, image):
        """
        Args:
            image (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            PIL: Transformed image.
        """
        left = randint(0, 200)
        top = randint(0, 200)
        image.paste(im=self.sign, box=(left, top), mask=self.mask)  # naklejenie maski na obrazek
        # box:  (left,upper) lub (left,upper,right,lower)
        # mask: - tylko wybrany region będzie update'owany; tam gdzie jest 0 nie będzie zmiany
        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'


def generate_transform(resolution=256):
    """
    Tworzy zestaw transformacji przygotowujących/randomizujących próbki.
    Jeśli `sign_filename` jest podane, to ten obrazek będzie nakładany na tło poprzednich.
    :return:
    """
    # podstawowe transformacje
    tts = [TS.Resize((resolution, resolution)),
           TS.RandomAffine(degrees=10, fill=(0, 0, 0), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0, 0.2)),
           TS.GaussianBlur(kernel_size=3),
           TS.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=(-0.1, 0.1)),
           TS.ToTensor()]

    return TS.Compose(tts)


def generate_sample(n_samples, image_dir, res, n_classes=2) -> tuple[tensor, tensor]:
    """
    dim = (batch,color,row,col)  dla tensora "samples" (return[0])
    """
    t = generate_transform(resolution=res)
    dataset = datasets.ImageFolder(image_dir, transform=t)
    dataloader = DataLoader(dataset, batch_size=15, shuffle=True)  # adjust to number of pictures
    samples, outputs = None, None
    while samples is None or samples.size()[0] < n_samples:
        for (images, classes) in dataloader:
            if samples is None:
                samples = images
                outputs = convert_index_to_bin_tensor(classes, n_classes)
            else:
                samples = torch.cat((samples, images), 0)
                outputs = torch.cat((outputs, convert_index_to_bin_tensor(classes, n_classes)), 0)
    return samples[:n_samples], outputs[:n_samples]


def generate_for_check(directory_name, resolution=256, n_classes=2):
    """
    Wczytuje obrazki z danego źródła i dostosowuje je (przez crop) do wybranej rozdzielczości.
    :param resolution: obrazki będą docięte do kwadratów res x res
    :param directory_name: źródło obrazków
    :param n_classes: liczba klas obrazków - czyli ile jest podfolderów typu 0,1,2...
    :return: Tensor typu [80, 3, 128, 128], i.e. nr. obrazka, nr koloru, rząd, kolumna
    """
    dataset = datasets.ImageFolder(directory_name, TS.Compose([TS.Resize((resolution, resolution)), TS.ToTensor()]))
    # tts = [TS.functional.crop()]    # fixme: use crop in TS.Compose
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    samples, outputs = None, None
    for (images, classes) in data_loader:
        if samples is None:
            samples = images
            outputs = convert_index_to_bin_tensor(classes, n_classes)
        else:
            samples = torch.cat((samples, images), 0)
            outputs = torch.cat((outputs, convert_index_to_bin_tensor(classes, n_classes)), 0)
        # for i in images:
        #     F.to_pil_image(i).show()
    return samples, outputs


if __name__ == '__main__':
    s, o = generate_sample(20, 'cars', res=128, n_classes=3)
    # s, o = generate_for_check('tcells', resolution=128, n_classes=3)
    print(s.shape)  # dim = (batch,color,row,col)
    print(s.shape)
    print(o)
