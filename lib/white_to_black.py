import configparser
import pickle, cv2, json
import numpy as np

with open("../MAGNETO_out/maldroid40-single/model_8x8_MI.json", 'rb') as f:
    model = json.load(f)

config = configparser.ConfigParser()
config.read('../config.ini')
from lib.load_dataset import load_dataset

dataset_param = config[config["SETTINGS"]["Dataset"]]
with open("../MAGNETO_out/maldroid40-single/train_8x8_MI.pickle", 'rb') as f:
    x_train = pickle.load(f)
with open("../MAGNETO_out/maldroid40-single/test_8x8_MI.pickle", 'rb') as f:
    x_test = pickle.load(f)
x_train = np.array(x_train)
x_test = np.array(x_test)
from matplotlib import pyplot as plt

new_train = np.zeros(x_train.shape)
new_test = np.zeros(x_test.shape)

zi = np.array(list(zip(model["xp"], model["yp"]))).astype(int) - 1
for i in range(len(x_train[0])):
    for j in range(len(x_train[0][i])):
        for k in zi:
            if i == k[0] and j == k[1]:
                print(i, j)
                new_train[:, i, j] = x_train[:, i, j]
                new_test[:, i, j] = x_test[:, i, j]

with open("../MAGNETO_out/maldroid40-single/train_8x8_MI_black.pickle", 'wb') as f:
    pickle.dump(new_train,f)
with open("../MAGNETO_out/maldroid40-single/test_8x8_MI_black.pickle", 'wb') as f:
    pickle.dump(new_test,f)

print(1)
