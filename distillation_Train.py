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
os.environ['HYPEROPT_FMIN_SEED'] = "9"

session = InteractiveSession(config=config_tf)

config = None
teacher = None
x_with_h = None
y_train = None
x_with_h_val = None
y_val = None
x_test = None
y_test = None
x_with_h_test = None

score_list = []
best_loss = None
now = datetime.now()
date = now.strftime("%Y-%m-%d-%H-%M-%S")
i = -1


def hyperopt_loop(param):
    global teacher
    global config
    global i
    i = i + 1
    shape = x_with_h.shape[2:5]

    student = load_student(config, shape, len(set(y_train)), param)
    student_compile(student, param)

    dashboard, wandb = set_dashboard_distiller(config, param, run_id=date, id=i)

    distiller = Distiller_heatmap(student=student, teacher=teacher)

    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=param["A"],
        temperature=param["T"],
    )

    callbacks = dashboard
    if config.getboolean("DISTILLATION", "EarlyStop"):
        stop_callback = tf.keras.callbacks.EarlyStopping(monitor='student_loss', min_delta=0.0001,
                                                         patience=config.getint("MODEL", "Patience"),
                                                         restore_best_weights=True, verbose=2)
        callbacks.append(stop_callback)

    one_hot_encode_y_train = to_categorical(y_train, num_classes=len(set(y_train)))
    one_hot_encode_y_val = to_categorical(y_val, num_classes=len(set(y_test)))
    if config.getboolean("DISTILLATION", "UseWeighedLoss"):

        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train),
                                                          y=y_train)
        sample_weight = np.zeros(shape=(len(y_train),))
        for e, cl in enumerate(class_weights):
            sample_weight[y_train == e] = cl

        history = distiller.fit(
            x=x_with_h,
            y=one_hot_encode_y_train,
            batch_size=param["batch"],
            epochs=config.getint("DISTILLATION", "Epochs"),
            validation_data=(x_with_h_val, one_hot_encode_y_val),
            verbose=1,
            callbacks=callbacks,
            sample_weight=sample_weight
        )
    else:
        history = distiller.fit(
            x=x_with_h,
            y=one_hot_encode_y_train,
            batch_size=param["batch"],
            epochs=config.getint("DISTILLATION", "Epochs"),
            validation_data=(x_with_h_val, one_hot_encode_y_val),
            verbose=1,
            callbacks=callbacks,
        )
    print("end")
    # distiller.evaluate(x_with_h, y_test)
    if len(set(y_train)) != 2:
        scores = check_score_and_save(history, distiller, x_with_h, y_train, x_with_h_val, y_val, x_with_h_test, y_test,
                                      config, save=False, distillation=True, dashboard=wandb)
    else:
        scores = check_score_and_save_bin(history, distiller, x_with_h, y_train, x_with_h_val, y_val, x_with_h_test,
                                          y_test, config,
                                          len(set(y_train)), 0,
                                          save=False, distillation=True, dashboard=wandb)
    scores.update(param)
    score_list.append(scores)
    if not os.path.exists(config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/distiller_" + str(date)):
        os.makedirs(config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/distiller_" + str(date))
    distiller.save_weights(
        config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/distiller_" + str(date) + "/" + str(i) + ".tf")
    global best_loss
    if best_loss is None:
        best_loss = score_list[-1]["val_student_loss"]
        distiller.save_weights(
            config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/distiller_" + str(date) + "/best.tf")
    elif score_list[-1]["val_student_loss"] < best_loss:
        best_loss = score_list[-1]["val_student_loss"]
        distiller.save_weights(
            config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/distiller_" + str(date) + "/best.tf")
    p = pd.DataFrame(score_list)
    p.to_excel(config[config["SETTINGS"]["Dataset"]]["OutputDir"] + "/distiller_" + str(date) + "/"+str(date)  + ".xlsx")
    K.clear_session()
    wandb.finish()
    return {'loss': scores["val_student_loss"], 'status': STATUS_OK}


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

    global x_with_h
    global y_train
    global x_with_h_val
    global y_val
    global teacher
    global x_test
    global y_test
    global x_with_h_test
    path = dataset_param["modelPath"]
    teacher = tf.keras.models.load_model(path, compile=False)  # vit

    x_train, y_train, x_test, y_test = load_dataset(dataset_param)
    x_train = get_rgb_images(x_train, x_train.shape[1:3])
    x_test = get_rgb_images(x_test, x_test.shape[1:3])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2,
                                                      random_state=config.getint("SETTINGS", "Seed"))

    im = []
    index = list(split(range(len(x_train)), 4))
    for k in index:
        im1, m1, _ = attention_map_no_norm_fast(teacher, x_train[k])
        im.extend(im1)
    im = np.array(im)
    if config["DISTILLATION"].getboolean("LAB"):
        im = color.rgb2lab(im / 255)
    # im[:,:,:,0]=minmax(im[:,:,:,0])
    # im[:,:,:,1]=minmax(im[:,:,:,1])
    # im[:,:,:,2]=minmax(im[:,:,:,2])

    im_val, m_val, _ = attention_map_no_norm_fast(teacher, x_val)
    if config["DISTILLATION"].getboolean("LAB"):
        im_val = color.rgb2lab(im_val / 255)
    # im_val[:, :, :, 0] = minmax(im_val[:, :, :, 0])
    # im_val[:, :, :, 1] = minmax(im_val[:, :, :, 1])
    # im_val[:, :, :, 2] = minmax(im_val[:, :, :, 2])

    im_test, m_test, _ = attention_map_no_norm_fast(teacher, x_test)
    if config["DISTILLATION"].getboolean("LAB"):
        im_test = color.rgb2lab(im_test / 255)

    # im_test[:, :, :, 0] = minmax(im_test[:, :, :, 0])
    # im_test[:, :, :, 0] = minmax(im_test[:, :, :, 0])
    # im_test[:, :, :, 1] = minmax(im_test[:, :, :, 1])
    # im_test[:, :, :, 2] = minmax(im_test[:, :, :, 2])

    x_with_h = np.array([x_train, im])
    x_with_h_val = np.array([x_val, im_val])
    x_with_h_test = np.array([x_test, im_test])

    x_with_h = np.swapaxes(x_with_h, 0, 1)
    x_with_h_val = np.swapaxes(x_with_h_val, 0, 1)
    x_with_h_test = np.swapaxes(x_with_h_test, 0, 1)

    # y_train = np.array([y_train, y_train])
    # y_val = np.array([y_val, y_val])

    trials = Trials()

    optimizable_variable = {"kernel": hp.choice("kernel", np.arange(2, 3 + 1)),
                            "filter": hp.choice("filter", [16, 32, 64, 128]),
                            "filter2": hp.choice("filter2", [16, 32, 64, 128]),
                            "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                            'dropout1': hp.uniform("dropout1", 0, 1),
                            'dropout2': hp.uniform("dropout2", 0, 1),
                            "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
                            "T": hp.choice("T", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                            "A": hp.uniform("A", 0, 1),
                            "epoch": 30}
    if config["SETTINGS"]["Dataset"] == "NSL":
        optimizable_variable = {"kernel": hp.choice("kernel", np.arange(2, 3 + 1)),
                                "filter": hp.choice("filter", [16, 32, 64, 128]),
                                "filter2": hp.choice("filter2", [16, 32, 64, 128]),
                                "batch": hp.choice("batch", [32, 64, 128, 256, 512]),
                                'dropout1': hp.uniform("dropout1", 0, 0.5),
                                'dropout2': hp.uniform("dropout2", 0, 0.5),
                                "learning_rate": hp.uniform("learning_rate", 1e-4, 1e-1),
                                "T": hp.choice("T", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
                                "A": hp.uniform("A", 0, 1)}

    fmin(hyperopt_loop, optimizable_variable, trials=trials, algo=tpe.suggest, max_evals=30)


if __name__ == '__main__':
    main()
