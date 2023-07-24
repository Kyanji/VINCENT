import configparser
import json
import os
from csv import DictReader

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from VINCENT_main import create_heatmap, create_ds_with_heatmap, VINCENT_fit
from lib.check_score import check_score_and_save
from lib.distiller_heatmap import Distiller_heatmap
from lib.load_dataset import load_dataset
from lib.load_student_model import load_student
from lib.set_dashboard import set_dashboard
from lib.to_rgb import get_rgb_images
from vit_main import vit_fit

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
    # SET ENVIRONMENTAL PARAMETERS
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

    # CONVERT TABULAR DATA TO IMAGES OR LOAD PICKLE
    if config.getboolean("SETTINGS", "UseMagnetoEncoding"):
        print("----USING MAGNETO ENCODING----")
        magneto_main(config, dataset_param, dataset_param.getboolean("toBinary"),
                     json.loads(dataset_param["toBinaryMap"]))

    print(dataset_param["OutputDirMagneto"] + dataset_param["trainName"])
    x_train, y_train, x_test, y_test = load_dataset(dataset_param)

    print("----SUMMARY VIT-TEACHER TRAINING----")
    print("XTRAIN SHAPE:\t", x_train.shape, "\tRANGE:\t", x_train.min(), x_train.max())
    print("XTEST SHAPE:\t", x_test.shape, "\tRANGE:\t", x_test.min(), x_test.max())
    print("CLASSES:\t", len(set(y_train)))

    if config.getboolean("SETTINGS", "UseRGBEncoding"):
        print("[+]RGB encoding")
        shape = x_train.shape[1:3]
        x_train = get_rgb_images(x_train, shape)
        x_test = get_rgb_images(x_test, shape)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2,
                                                      random_state=config.getint("SETTINGS", "Seed"))
    if config.getboolean("SETTINGS", "TrainVIT"):
        # TRAIN VIT
        if config.getboolean("SETTINGS", "Wandb"):
            dashboard, wandb = set_dashboard(config)
        else:
            dashboard = []
            wandb = None
        teacher, history = vit_fit(config, x_train, y_train, x_val, y_val, dashboard)
        scores, res_test = check_score_and_save(history, teacher, x_train, y_train, x_val, y_val, x_test, y_test,
                                                config, wandb)
        print(scores)
        cr_teacher = classification_report(y_test, res_test)
        print(cr_teacher)

    else:
        path = dataset_param["VIT_Teacher_Path"]
        teacher = tf.keras.models.load_model(path, compile=False)  # vit

    im = create_heatmap(teacher, x_train, 1)
    im_val = create_heatmap(teacher, x_val, 1)
    im_test = create_heatmap(teacher, x_test, 1)

    # CRAFT ARRAY WITH ORIGINAL DATA AND ORIGINAL+HEATMAP
    x_with_h = create_ds_with_heatmap(x_train, im)
    x_with_h_val = create_ds_with_heatmap(x_val, im_val)
    x_with_h_test = create_ds_with_heatmap(x_test, im_test)

    if config.getboolean("SETTINGS", "TrainVINCENT"):
        distiller, score = VINCENT_fit(config, teacher, x_with_h, y_train, x_with_h_val, y_val, x_with_h_test, y_test)
    else:
        if "tf" in dataset_param["VINCENTPath"]:
            with open(dataset_param["VINCENTPath"].replace(dataset_param["VINCENTPath"].split(".")[-1],
                                                                 "csv"), 'r') as f:
                dict_reader = DictReader(f)
                hyp = list(dict_reader)[0]
        hyp["dropout1"] = float(hyp["dropout1"])
        hyp["dropout2"] = float(hyp["dropout2"])
        hyp["kernel"] = int(hyp["kernel"])
        student = load_student(config, input_shape=x_train.shape[1: 4], hyperparameters=hyp,
                               num_classes=len(set(y_train)))
        distiller = Distiller_heatmap(student=student, teacher=teacher)
        distiller.load_weights(dataset_param["VINCENTPath"])

    y_pred = distiller.predict(x_with_h_test)
    y_pred = np.argmax(y_pred, axis=-1)
    cr_student = classification_report(y_test, y_pred)
    print(cr_student)
    print("-----")


if __name__ == '__main__':
    main()
