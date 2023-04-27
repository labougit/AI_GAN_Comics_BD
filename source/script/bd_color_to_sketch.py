from keras.models import load_model
import cv2
import numpy as np
from helper import *

import os
from os.path import isfile, join
import copy


mod = load_model('./mod.h5')

def get(path, new_path):
    from_mat = cv2.imread(path)
    width = float(from_mat.shape[1])
    height = float(from_mat.shape[0])
    new_width = 0
    new_height = 0
    if (width > height):
        from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512
    #cv2.imshow('raw', from_mat)
    #cv2.imwrite('raw.jpg',from_mat)
    new_from_mat = copy.deepcopy(from_mat)
    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    #show_active_img_and_save('sketchKeras_colored', line_mat, 'sketchKeras_colored.jpg')
    line_mat = np.amax(line_mat, 2)
    #show_active_img_and_save_denoise_filter2('sketchKeras_enhanced', line_mat, 'sketchKeras_enhanced.jpg')
    #show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat, 'sketchKeras_pured.jpg')
    show_active_img_and_save_denoise('sketchKeras', line_mat, new_path, new_from_mat)
    #cv2.waitKey(0)
    return


path_folder = "../../Sketch_example/sketchs/"
new_path_folder = "../../Sketch_example/sketchs_resized/"
count = 0
for file in os.listdir(path_folder):
    file_path = join(path_folder, file)

    extension = file[-4:]
    if isfile(file_path) and count < 11 and (extension == ".jpg" or extension == ".png"):
        print("THE FILE PATH IS: ", file_path)
        

        new_file_path = join(new_path_folder, file)
        get(file_path, new_file_path)
    #count +=1
