import csv
import json
import pickle

import numpy as np
import pandas as pd
from keras import Model
from keras.saving.save import load_model

from MAGNETO.lib import VecToImage


def magneto_main(config, dataset_param, toBinary, toBinaryMap):
    with open(dataset_param["tabular_dataset_path"] + dataset_param["tabular_trainfile"], 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))), "class": dataset_param["classes"]}
        data["Classification"] = data["Xtrain"][dataset_param["classification"]]
        del data["Xtrain"][dataset_param["classification"]]
    with open(dataset_param["tabular_dataset_path"] + dataset_param["tabular_testfile"], 'r') as file:
        Xtest = pd.DataFrame(list(csv.DictReader(file)))
        Xtest.replace("", np.nan, inplace=True)
        Xtest.dropna(inplace=True)
        data["Xtest"] = Xtest
        data["Ytest"] = data["Xtest"][dataset_param["classification"]]
        del data["Xtest"][dataset_param["classification"]]

    # CALCULATE MI on binary label
    y1 = data["Classification"].astype(float)
    y2 = data["Ytest"].astype(float)

    data["Xtrain"] = data["Xtrain"].astype(float)
    data["Xtest"] = data["Xtest"].astype(float)

    # if y1.min() != 0:
    #     y1 = y1 - 1
    #     y2 = y2 - 1

    print("[+]Mapping To Binary")
    f_myfile = open(config[config["SETTINGS"]["Dataset"]]["OutputDirMagneto"] + 'Ytrain.pickle', 'wb')
    pickle.dump(y1, f_myfile)
    f_myfile.close()

    f_myfile = open(config[config["SETTINGS"]["Dataset"]]["OutputDirMagneto"] + 'Ytest.pickle', 'wb')
    pickle.dump(y2, f_myfile)
    f_myfile.close()

    data["Classification"] = data["Classification"].map(toBinaryMap)
    data["Ytest"] = data["Ytest"].map(toBinaryMap)

    XGlobal, YGlobal, XTestGlobal, YTestGlobal = VecToImage.toImage(config, data, norm=True)
    return XGlobal, y1, XTestGlobal, y2
