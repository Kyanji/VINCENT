import configparser
import os
import pickle
from datetime import datetime

import cv2
import json
import numpy as np
from skimage import color, io

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from keras import Input
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.utils import class_weight
from vit_keras import vit, utils, visualize

from lib.check_score import check_score_and_save, check_score_and_save_bin
from lib.distiller_heatmap import Distiller_heatmap
from lib.load_dataset import load_dataset
from lib.load_model import load_model
from lib.load_student_model import load_student
from lib.model_compile import model_compile
from lib.set_dashboard import set_dashboard, set_dashboard_distiller
from lib.student_compile import student_compile
from lib.to_rgb import get_rgb_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from lib.custom_attention import attention_map_no_norm, attention_map_no_norm_fast
from keras_cv_attention_models import visualizing, test_images, botnet, halonet, beit, levit, coatnet, coat
from tensorflow.keras import layers
import pandas as pd
from keras import backend as K

tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)
config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

# os.environ['HYPEROPT_FMIN_SEED'] = "9"
os.environ['HYPEROPT_FMIN_SEED'] = "0"

session = InteractiveSession(config=config_tf)

config = None
x_train = None
y_train = None
x_val = None
y_val = None
x_test = None
y_test = None

score_list = []
best_loss = None
now = datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M-%S")
i = -1


def hyperopt_loop(param):
    global x_train, x_val, x_test
    global y_train, y_val, y_test
    global i
    i = i + 1
    shape = x_train.shape[1:4]
    start = datetime.now()
    model = load_student(config, shape, len(set(y_train)), param)
    student_compile(model, param)
    if config.getboolean("SETTINGS", "Wandb"):
        dashboard, wandb = set_dashboard_distiller(config, param, run_id=date, id=i)
    else:
        dashboard = []
        wandb = None

    callbacks = dashboard
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                                     patience=config.getint("DISTILLATION", "Patience"),
                                                     restore_best_weights=True, verbose=2)
    callbacks.append(stop_callback)

    one_hot_encode_y_train = to_categorical(y_train, num_classes=len(set(y_train)))
    one_hot_encode_y_val = to_categorical(y_val, num_classes=len(set(y_test)))

    history = model.fit(
        x=x_train,
        y=one_hot_encode_y_train,
        batch_size=param["batch"],
        epochs=config.getint("DISTILLATION", "Epochs"),
        validation_data=(x_val, one_hot_encode_y_val),
        verbose=1,
        callbacks=callbacks,
    )
    print("end")
    # distiller.evaluate(x_with_h, y_test)
    if len(set(y_train)) != 2:
        scores = check_score_and_save(history, model, x_train, y_train, x_val, y_val, x_test, y_test,
                                      config, save=False, distillation=False, dashboard=wandb,
                                      time=datetime.now() - start)
    else:
        scores = check_score_and_save_bin(history, model, x_train, y_train, x_val, y_val, x_test,
                                          y_test, config,
                                          len(set(y_train)), 0,
                                          save=False, distillation=False, dashboard=wandb, time=datetime.now() - start)
    scores.update(param)
    score_list.append(scores)
    if not os.path.exists(config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date)):
        os.makedirs(config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date))
    model.save_weights(
        config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date) + "/" + str(i) + ".tf")
    global best_loss
    if best_loss is None:
        best_loss = score_list[-1]["val_loss"]
        model.save_weights(
            config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date) + "/best.tf")
    elif score_list[-1]["val_loss"] < best_loss:
        best_loss = score_list[-1]["val_loss"]
        model.save_weights(
            config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date) + "/best.tf")
    p = pd.DataFrame(score_list)
    p.to_excel(
        config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/cnn" + str(date) + "/" + str(date) + ".xlsx")
    K.clear_session()
    if wandb is not None:
        wandb.finish()
    return {'loss': scores["val_loss"], 'status': STATUS_OK}


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def minmax(X, min=0, max=255):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def main():
    global config
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

    global x_train
    global y_train
    global x_val
    global y_val
    global x_test
    global y_test

    x_train, y_train, x_test, y_test = load_dataset(dataset_param)
    x_train = get_rgb_images(x_train, x_train.shape[1:3])
    x_test = get_rgb_images(x_test, x_test.shape[1:3])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2,
                                                      random_state=config.getint("SETTINGS", "Seed"))

    trials = Trials()

    optimizable_variable = {"kernel": hp.choice("kernel", np.arange(2, 3 + 1)),

                            "batch": hp.choice("batch", [64, 128, 256, 512]),
                            'dropout1': hp.uniform("dropout1", 0, 1),
                            'dropout2': hp.uniform("dropout2", 0, 1),
                            "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
                            }
    if config["SETTINGS"]["Dataset"] == "NSL":
        optimizable_variable = {"kernel": hp.choice("kernel", np.arange(2, 3 + 1)),

                                "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                                'dropout1': hp.uniform("dropout1", 0, 0.5),
                                'dropout2': hp.uniform("dropout2", 0, 0.5),
                                "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
                                }

    fmin(hyperopt_loop, optimizable_variable, trials=trials, algo=tpe.suggest, max_evals=20)


if __name__ == '__main__':
    main()
