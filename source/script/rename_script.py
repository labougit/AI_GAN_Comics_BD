import os
from os.path import isfile, join

def new_name(count):
    if (count/10000 >= 1.0):
        return (str(count))
    if (count/1000 >= 1.0):
        return ("0"+str(count))
    if (count/100 >= 1.0):
        return ("00"+str(count))
    if (count/10 >= 1.0):
        return ("000"+str(count))
    return ("0000"+str(count))

count = 1

path_folder = "./train/"
for file in os.listdir(path_folder):
    file_path = join(path_folder, file)
    if isfile(file_path):
        extension = file[-4:]
        file_name = new_name(count)+extension
        new_file_name=join(path_folder, file_name)
        count +=1
        os.rename(file_path, new_file_name)


