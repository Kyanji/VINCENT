import configparser
import os
import pickle
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from vit_keras import vit, utils, visualize

from lib.distiller_heatmap import Distiller_heatmap
from lib.load_dataset import load_dataset
from lib.load_model import load_model
from lib.model_compile import model_compile
from lib.to_rgb import get_rgb_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from lib.custom_attention import attention_map_no_norm
from keras_cv_attention_models import visualizing, test_images, botnet, halonet, beit, levit, coatnet, coat
from tensorflow.keras import layers

tf.config.run_functions_eagerly(True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)

config = configparser.ConfigParser()
config.read('config.ini')
os.environ['PYTHONHASHSEED'] = config["SETTINGS"]["Seed"]
tf.compat.v1.set_random_seed(config["SETTINGS"]["Seed"])

# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
# os.environ['PYTHONHASHSEED'] = config["SETTINGS"]["Seed"]
np.random.seed(int(config["SETTINGS"]["Seed"]))
# rn.seed(1254)
tf.keras.utils.set_random_seed(int(config["SETTINGS"]["Seed"]))
dataset_param = config[config["SETTINGS"]["Dataset"]]

x_train, y_train, x_test, y_test = load_dataset(dataset_param)
x_train = get_rgb_images(x_train[0:200], x_train.shape[1:3])
x_test = get_rgb_images(x_test[0:200], x_train.shape[1:3])
y_train = y_train[0:200]
shape = x_train[0].shape[0]
path = config["VISUALIZE"]["modelPath"]
teacher = tf.keras.models.load_model(path, compile=False)  # vit
# teacher.predict(x_train[0].reshape(1, 12, 12, 3))


student = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(shape, shape, 3)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.ReLU(alpha=0.2),
        # layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(5),  # TODO
        layers.Dense(5),
    ],
    name="student",
)
student.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], run_eagerly=True
)
# student.fit(x_train, y_train, epochs=3)

distiller = Distiller_heatmap(student=student, teacher=teacher)

distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

distiller.fit(x_train, y_train, epochs=10)
distiller.evaluate(x_test, y_test)
