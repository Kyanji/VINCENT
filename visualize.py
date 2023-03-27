import configparser
import os
import pickle
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from vit_keras import vit, utils, visualize
from lib.load_dataset import load_dataset
from lib.load_model import load_model
from lib.model_compile import model_compile
from lib.to_rgb import get_rgb_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from lib.custom_attention import attention_map_no_norm
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

path = config["VISUALIZE"]["modelPath"]
model = tf.keras.models.load_model(path, compile=False)  # vit

# model = tf.keras.models.load_model("./res/malmem/2023-03-10-14-54-28.h5", compile=False)
shape = x_train.shape[1:3]

classes = {}
rgb_train = []
fig, axn = plt.subplots(ncols=3, nrows=len(set(y_train)))
i = 0
attention_mapl = []
maskl = {}
import pandas as pd
df=[]
classes_list=[]
for y in list(set(y_train)):
    classes[y] = np.where(y_train == y)[0][0]
    img = get_rgb_images([x_train[classes[y]]], shape)[0]
    rgb_train.append(img)
    attention_map, mask = attention_map_no_norm(model=model, image=rgb_train[-1])
    a=pd.DataFrame(mask.reshape(mask.shape[1],mask.shape[1]))
    #a.to_excel(str(y)+".xlsx")
    classes_list.append(str(y))
    df.append(a)
    maskl[str(y)] = str(mask)
    axn[i][0].axis('off')
    axn[i][1].axis('off')
    axn[i][2].axis('off')
    axn[i][0].set_title('Original')
    axn[i][1].set_title('Attention Map')
    axn[i][2].set_title('Map')
    _ = axn[i][0].imshow(img)
    _ = axn[i][1].imshow(attention_map)
    _ = axn[i][2].imshow(mask)
    i = i + 1
plt.show()

writer = pd.ExcelWriter("./visualize.xlsx", engine='xlsxwriter')
for i in range(len(df)):

    df[i].to_excel(writer, sheet_name=classes_list[i])

writer.save()
writer.close()



# x_test = get_rgb_images(x_test, shape)
# _ = visualizing.plot_attention_score_maps(model, x_train[0], rescale_mode='torch',attn_type="bot")
