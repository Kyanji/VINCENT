import configparser
import os
from datetime import datetime

import pickle
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from lib.check_score import check_score_and_save, check_score_and_save_bin
from lib.fit import fit
from lib.load_dataset import load_dataset
from lib.load_model import load_model
from lib.model_compile import model_compile
from lib.set_dashboard import set_dashboard
from lib.to_rgb import rgb_image_generation, get_rgb_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from MAGNETO.magneto_main import magneto_main

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    os.environ['PYTHONHASHSEED'] = config["SETTINGS"]["Seed"]
    tf.compat.v1.set_random_seed(config["SETTINGS"]["Seed"])

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    os.environ['PYTHONHASHSEED'] = config["SETTINGS"]["Seed"]
    np.random.seed(int(config["SETTINGS"]["Seed"]))
    # rn.seed(1254)
    tf.keras.utils.set_random_seed(int(config["SETTINGS"]["Seed"]))

    dataset_param = config[config["SETTINGS"]["Dataset"]]

    if config.getboolean("SETTINGS", "UseMagnetoEncoding"):
        magneto_main(config, dataset_param, dataset_param.getboolean("toBinary"),
                     json.loads(dataset_param["toBinaryMap"]))

    x_train, y_train, x_test, y_test = load_dataset(dataset_param)
    print("----SUMMARY----")
    print("XTRAIN SHAPE:\t", x_train.shape, "\tRANGE:\t", x_train.min(), x_train.max())
    print("XTEST SHAPE:\t", x_train.shape, "\tRANGE:\t", x_train.min(), x_train.max())
    print("CLASSES:\t", len(set(y_train)))
    if config.getboolean("SETTINGS", "UseRGBEncoding"):
        print("[+]RGB encoding")
        shape = x_train.shape[1:3]
        x_train = get_rgb_images(x_train, shape)
        x_test = get_rgb_images(x_test, shape)

    if config.getboolean("SETTINGS", "UseScale0_1"):
        print("[+]0 1 norm")
        x_train = np.array(x_train) / 255
        x_test = np.array(x_test) / 255
    if config.getboolean("SETTINGS", "UseScale-1_1"):
        print("[+]-1 1 norm")
        x_train = np.array(x_train) / 255
        x_test = np.array(x_test) / 255
        x_train = x_train * 2 - 1
        x_test = x_test * 2 - 1
    if config.getboolean("SETTINGS", "Resize"):
        print("[+]Resizing to ", json.loads(config["SETTINGS"]["ResizeShape"]))

        new_size = json.loads(config["SETTINGS"]["ResizeShape"])

        x_train_resized = []
        x_test_resized = []
        for i in x_train:
            xx = cv2.resize(i, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)
            x_train_resized.append(xx)
        for i in x_test:
            xx = cv2.resize(i, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)
            x_test_resized.append(xx)
        x_train_resized = np.array(x_train_resized)
        x_test_resized = np.array(x_test_resized)

    if config.getboolean("SETTINGS", "Resize"):
        x_train, x_val, y_train, y_val = train_test_split(x_train_resized, y_train, stratify=y_train, test_size=0.2,
                                                          random_state=config.getint("SETTINGS", "Seed"))
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2,
                                                          random_state=config.getint("SETTINGS", "Seed"))

    model = load_model(config, x_train.shape[1:4], len(set(y_train)))
    model = model_compile(model, config)
    dashboard, wandb = set_dashboard(config)

    print("----TRAINING SUMMARY----")
    print("SHAPE:\t", x_train.shape, "\tRANGE:\t", x_train.min(), x_train.max())
    print("CLASSES:\t", len(set(y_train)))
    start = datetime.now()
    model, history = fit(model, config, x_train, y_train, x_val, y_val, dashboard)
    end = datetime.now()

    if config.getboolean("SETTINGS", "Resize"):
        scores = check_score_and_save_bin(history, model, x_train, y_train, x_val, y_val, x_test_resized, y_test,
                                          config, len(set(y_train)), end - start,
                                          wandb)
    else:
        scores = check_score_and_save_bin(history, model, x_train, y_train, x_val, y_val, x_test, y_test, config,
                                          len(set(y_train)), end - start,
                                          wandb)

    print("-----")


if __name__ == '__main__':
    main()
