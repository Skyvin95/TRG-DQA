import torchvision.transforms.functional as TF
from PIL import Image
import random


def resize_image(image):

    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if aspect_ratio < 1:
        new_width = 384
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 384
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))

    return resized_image, new_width, new_height


def resize_crop(raw_image, enhance_image):

    raw_img, _, _ = resize_image(raw_image)
    enhance_img, new_width, new_height = resize_image(enhance_image)

    left = random.randint(0, new_width - 384)
    top = random.randint(0, new_height - 384)
    random_crop_raw = TF.crop(raw_img, top, left, 384, 384)
    random_crop_enhance = TF.crop(enhance_img, top, left, 384, 384)

    return random_crop_raw, random_crop_enhance