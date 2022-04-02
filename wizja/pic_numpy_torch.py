import torch
from PIL import Image, ImageOps
from PIL.Image import BICUBIC
import numpy as np

import torchvision.transforms as T

img = Image.open('pic.jpg')
img = img.resize((150, 150), resample=BICUBIC)
dat = np.asarray(img)
print(dat.shape)  # (30,30,3)
dat = np.swapaxes(dat, 0, -1)
print(dat.shape)  # (3, 10, 10)
print(dat[:, :, 0])
print('---')
# print(dat[:,:,1])
# print('---')
# print(dat[:,:,2])
# img.show()
# img1 = Image.fromarray(dat[:, :, 0])  # , 'RGB'  lub 'RGBA' jeśli jest alpha, lub pominąć jeśli jest czarno-biały
# img1.show()

# x_np = torch.from_numpy(dat[:, :, 0])
# x_np = x_np.double()
# print(x_np)

affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
affine_imgs = [affine_transfomer(img) for _ in range(4)]
for im in affine_imgs:
    im.show()