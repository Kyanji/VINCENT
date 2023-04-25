import configparser
import os
import pickle
import cv2
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import color
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

from vit_keras import vit, utils, visualize
from lib.load_dataset import load_dataset
from lib.load_model import load_model
from lib.model_compile import model_compile
from lib.to_rgb import get_rgb_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from lib.custom_attention import attention_map_no_norm, attention_map_no_norm_fast
from keras_cv_attention_models import visualizing, test_images, botnet, halonet, beit, levit, coatnet, coat

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)

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

x_train, y_train, x_test, y_test = load_dataset(dataset_param)
x_train = get_rgb_images(x_train, x_train.shape[1:3])
x_test = get_rgb_images(x_test, x_test.shape[1:3])

path = dataset_param["modelPath"]
teacher = tf.keras.models.load_model(path, compile=False)  # vit

fig, axn = plt.subplots(ncols=5, nrows=len(set(y_train)))

pred_train = teacher.predict(x_train)
pred_train = np.argmax(pred_train, axis=-1)
cm_train = metrics.confusion_matrix(y_train, pred_train)

pred_test = teacher.predict(x_test)
pred_test = np.argmax(pred_test, axis=-1)

cm_test = metrics.confusion_matrix(y_test, pred_test)

df = pd.DataFrame(cm_train)
df_conf = pd.DataFrame(cm_test)
# df.to_excel(, index=False)

writer = pd.ExcelWriter("./"+config["SETTINGS"]["Dataset"] + "_cm.xlsx", engine='xlsxwriter')
df.to_excel(writer, sheet_name='results')
df_conf.to_excel(writer, sheet_name='configuration')
writer.save()
writer.close()

k = 0
im1, m1,scaled = attention_map_no_norm_fast(teacher, x_train)
lab_im1 = []
for i in im1:
    lab_im1.append(color.rgb2lab(i / 255))
lab_im1 = np.array(lab_im1)

for i in list(set(y_train)):
    b = np.where(y_train == i)

    #  for k in im1[b]:
    #     v = cv2.resize(k, (800, 800), interpolation=cv2.INTER_NEAREST)
    #     cv2.imwrite( "test/" + str(pp) + ".png",cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    #     pp = pp + 1
    # plt.imsave(str(i) + ".png", cv2.resize(np.mean(m1, axis=0).reshape(8, 8),(500,500),interpolation=cv2.INTER_NEAREST  ),)

    axn[k][0].axis('off')
    axn[k][1].axis('off')
    axn[k][2].axis('off')
    axn[k][0].axis('off')
    axn[k][1].axis('off')
    axn[k][2].axis('off')
    axn[k][0].set_title('original')
    axn[k][1].set_title('M Heat')
    axn[k][2].set_title('M Heat Lab')
    axn[k][3].set_title('scaled')
    axn[k][4].set_title('mask')
    _ = axn[k][0].imshow(np.mean(x_train,axis=0).astype(np.uint8))
    _ = axn[k][1].imshow(np.mean(im1[b], axis=0).astype(np.uint8))
    m = np.mean(lab_im1[b], axis=0).astype(np.uint8)
    _ = axn[k][2].imshow(m)
    _ = axn[k][3].imshow(np.mean(scaled[b], axis=0).astype(np.uint8))
    _ = axn[k][4].imshow(np.mean(m1[b], axis=0))
    k = k + 1

    # plt.show()

    print(i)
plt.show()

fig, axn = plt.subplots(ncols=5, nrows=len(set(y_train)))

im1, m1,scaled = attention_map_no_norm_fast(teacher, x_test)
lab_im1 = []
for i in im1:
    lab_im1.append(color.rgb2lab(i / 255))
lab_im1 = np.array(lab_im1)
k = 0
for i in list(set(y_test)):
    b = np.where(y_test == i)

    #  for k in im1[b]:
    #     v = cv2.resize(k, (800, 800), interpolation=cv2.INTER_NEAREST)
    #     cv2.imwrite( "test/" + str(pp) + ".png",cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
    #     pp = pp + 1
    # plt.imsave(str(i) + ".png", cv2.resize(np.mean(m1, axis=0).reshape(8, 8),(500,500),interpolation=cv2.INTER_NEAREST  ),)

    axn[k][0].axis('off')
    axn[k][1].axis('off')
    axn[k][2].axis('off')
    axn[k][0].set_title('original')
    axn[k][1].set_title('M Heat')
    axn[k][2].set_title('M Heat Lab')
    axn[k][3].set_title('scaled')
    axn[k][4].set_title('mask')
    _ = axn[k][0].imshow(np.mean(x_test,axis=0).astype(np.uint8))
    _ = axn[k][1].imshow(np.mean(im1[b], axis=0).astype(np.uint8))
    m = np.mean(lab_im1[b], axis=0).astype(np.uint8)
    _ = axn[k][2].imshow(m)
    _ = axn[k][3].imshow(np.mean(scaled[b], axis=0).astype(np.uint8))
    _ = axn[k][4].imshow(np.mean(m1[b], axis=0))
    k = k + 1

    # plt.show()

    print(i)
plt.show()
