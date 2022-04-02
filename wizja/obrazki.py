from PIL import Image
from PIL.Image import BICUBIC

img = Image.open('pic.jpg')
img = img.resize((64,64), resample=BICUBIC)
img.show()