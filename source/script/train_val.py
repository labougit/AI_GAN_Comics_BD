import os
from os.path import isfile, join
import shutil

path_folder = "./train_sketch/"
path_destination = "./val_bd/"
count = 0

for file in os.listdir(path_folder):
    file_path = join(path_folder, file)
    if isfile(file_path) and count%5 == 0:
        shutil.move(file_path, join(path_destination, file))
    count += 1

        