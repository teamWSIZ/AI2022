from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import torchvision.transforms as T

# pip install Pillow

img = Image.open('pic.jpg')
img = img.resize((120,120))
# img.show()
tablica = np.asarray(img) #(350, 400, 3) --- wysokość, szerokość, nr. koloru (0,1,2 → RGB)
print(tablica.shape)
print(tablica[:,:,0])
dat = np.swapaxes(tablica, 0, -1)
print(dat.shape)    # (3, 400, 350) --- kolor, szerokość, wysokość
print(dat[0]) # kanał "czerwony"

# zamiana "danych" -- czyli tablic numpy lub tensorów torch-a na obrazki PIL-a
# kanal_czerwony = Image.fromarray(dat[0])
# kanal_czerwony.show()
# Image.fromarray(dat[1]).show()
# Image.fromarray(dat[2]).show()

affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
blur_transfomer = T.GaussianBlur(17, sigma=(1.2,1.5))
images = [blur_transfomer(affine_transfomer(img)) for _ in range(6)]
for im in images:
    im.show()