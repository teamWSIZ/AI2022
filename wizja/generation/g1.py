from random import randint

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as TS
import torchvision.transforms.functional as F


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
           TS.RandomAffine(degrees=10, fill=(0, 0, 0), translate=(0.1, 0.1), scale=(0.5, 1.5), shear=(0, 0.2)),
           TS.GaussianBlur(kernel_size=3),
           TS.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=(-0.2,0.2)),
           TS.ToTensor()]

    return TS.Compose(tts)


def generate_sample(img_count, directory_name, res):
    """
    :return: Tensor typu [80, 3, 128, 128], i.e. nr. obrazka, nr koloru, rząd, kolumna

    Uwaga: liczba obrazków zawsze jest wielokrotnością liczby obrazków w folderze z których są pobierane.
    """
    t = generate_transform(resolution=res)
    dataset = datasets.ImageFolder(directory_name, transform=t)
    dataloader = DataLoader(dataset, batch_size=15, shuffle=True)  # adjust to number of pictures
    res = None
    while res is None or res.size()[0] < img_count:
        for (images, classes) in dataloader:
            # wizualne sprawdzanie wyników
            # for i in images:
            #     F.to_pil_image(i).show()
            if res is None:
                res = images
            else:
                res = torch.cat((res, images), 0)
    return res


if __name__ == '__main__':
    generate_sample(50, 'tcells', res=128)
