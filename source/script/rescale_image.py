import os
from os.path import isfile, join

from PIL import Image

from torchvision import transforms

def addPad(image):
    size = 512

    image = Image.open(image)
    image_width, image_height = image.size
    imageA = image.crop((0, 0, image_width // 2, image_height))
    imageB = image.crop((image_width // 2, 0, image_width, image_height))

    # default transforms, pad if needed and center crop 512
    width_pad = size - image_width // 2
    if width_pad < 0:
        # do not pad
        width_pad = 0

    height_pad = size - image_height
    if height_pad < 0:
        height_pad = 0

    # padding as white
    padding = transforms.Pad((width_pad // 2, height_pad // 2 + 1,
                                width_pad // 2 + 1, height_pad // 2),
                                (255, 255, 255))

    # use center crop
    crop = transforms.CenterCrop(size)

    imageA = padding(imageA)
    imageA = crop(imageA)

    imageB = padding(imageB)
    imageB = crop(imageB)

    new_image = Image.new('RGB', (imageA.width + imageB.width, imageA.height))
    new_image.paste(imageA, (0, 0))
    new_image.paste(imageB, (imageA.width, 0))
    return new_image

path_folder = "../test_luke/sketch"
for file in os.listdir(path_folder):
    file_path = join(path_folder, file)
    if isfile(file_path):
        addPad(file_path).save(file_path)
