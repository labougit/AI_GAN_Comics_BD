import os
from os.path import isfile, join

from PIL import Image

import colorgram
import json

path_folder = "./train_sketch/"
path_destination = "./colorgram_bd/"

count = 0
for file in os.listdir(path_folder):
    file_path = join(path_folder, file)
    if isfile(file_path):
        # count += 1
        image = Image.open(file_path)
        image_width, image_height = image.size
        imageA = image.crop((0, 0, image_width // 2, image_height))
        imageB = image.crop((image_width // 2, 0, image_width, image_height))

        colors = colorgram.extract(imageA, 16)
        print("File path: ", file_path)
        print("Lenght colors: ", len(colors))
        color_list = {}
        for i in range(4):
            color_list[str(i+1)] = {}
            for j in range(4):
                color_list[str(i+1)][str(j+1)] = colors[(4*i+j)%len(colors)].rgb
        with open(join(path_destination, file[:-4]+".json"), "w") as f:
            f.write(json.dumps(color_list))
    else:
        break
        