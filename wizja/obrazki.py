from PIL import Image, ImageOps, ImageEnhance
from PIL.Image import BICUBIC

# pip install Pillow

img = Image.open('pic.jpg')
igs = ImageOps.grayscale(img)
# bright = ImageEnhance.Brightness(img).enhance(1.5)
enhanced_image = ImageEnhance.Sharpness(igs).enhance(1.5).resize((1024,1024))
# img = img.resize((64,64), resample=BICUBIC)
enhanced_image.show()