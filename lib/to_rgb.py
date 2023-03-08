import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse


def dec_to_bin(x):
    return format(int(x), "b")


def rgb_image_generation(df):
    list_image_flat = []
    size, padding = get_image_size(64)
    vec = [[0, 0, 0]] * padding
    for i in range(len(df)):
        for j in range(len(df[i])):
            list_image = []
            if df[i][j] > 1:
                c = 1.0
            elif df[i][j] < 0:
                c = 0.0
            else:
                c = df[i][j]
            v = c * (2 ** 24 - 1)
            bin_num = dec_to_bin(int(v))
            if len(bin_num) < 24:
                pad = 24 - len(bin_num)
                zero_pad = "0" * pad
                line = zero_pad + str(bin_num)
                n = 8
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
                int_num = [int(element, 2) for element in rgb]
            else:
                n = 8
                line = str(bin_num)
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
                int_num = [int(element, 2) for element in rgb]
            list_image_flat.append(int_num)


    return list_image_flat


def get_image_size(num_col):
    import math
    matx = round(math.sqrt(num_col))
    if num_col > (matx * matx):
        matx = matx + 1
        padding = (matx * matx) - num_col
    else:
        padding = (matx * matx) - num_col
    return matx, padding


def get_rgb_images(images, shape):
    rgb_images = []
    for i in images:
        rgb_images.append(np.array(rgb_image_generation(i)).reshape(shape[0], shape[1], 3))

    rgb_images = np.array(rgb_images)
    return rgb_images
