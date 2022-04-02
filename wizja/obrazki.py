from PIL import Image, ImageOps, ImageEnhance
from PIL.Image import BICUBIC

# pip install Pillow

img = Image.open('pic.jpg')
igs = ImageOps.grayscale(img)
# bright = ImageEnhance.Brightness(img).enhance(1.5)
bright = ImageEnhance.Sharpness(img).enhance(1.5)
# img = img.resize((64,64), resample=BICUBIC)
bright.show()