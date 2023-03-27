import csv
import json
import pickle

import numpy as np
import pandas as pd
from keras import Model
from keras.saving.save import load_model

from MAGNETO.lib import VecToImage


def magneto_main(config, dataset_param, toBinary, toBinaryMap):
    with open(dataset_param["path"] + dataset_param["trainFile"], 'r') as file:
        data = {"Xtrain": pd.DataFrame(list(csv.DictReader(file))), "class": dataset_param["classes"]}
        data["Classification"] = data["Xtrain"][dataset_param["classification"]]
        del data["Xtrain"][dataset_param["classification"]]
    with open(dataset_param["path"] + dataset_param["testFile"], 'r') as file:
        Xtest = pd.DataFrame(list(csv.DictReader(file)))
        Xtest.replace("", np.nan, inplace=True)
        Xtest.dropna(inplace=True)
        data["Xtest"] = Xtest
        data["Ytest"] = data["Xtest"][dataset_param["classification"]]
        del data["Xtest"][dataset_param["classification"]]

    if config.getboolean("MAGNETO", "No_0_MI"):
        with open(config["MAGNETO"]["No_0_MI_path"]) as json_file:
            j = json.load(json_file)
        data["Xtrain"] = data["Xtrain"].drop(columns=j)
        data["Xtest"] = data["Xtest"].drop(columns=j)
        print("0 MI features dropped!")

    if config.getboolean("MAGNETO", "autoencoder"):
        autoencoder = load_model(config["MAGNETO"]["autoencoderPath"])
        autoencoder.summary()
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encod2').output)
        encoder.summary()
        # usa l'encoder con predict sul train_X e poi su test_X. Io qui ho creato anche il dataframe per salvarlo poi
        # come csv
        encoded_train = pd.DataFrame(encoder.predict(data["Xtrain"]))
        data["Xtrain"] = encoded_train.add_prefix('feature_')
        encoded_test = pd.DataFrame(encoder.predict(data["Xtest"]))
        data["Xtest"] = encoded_test.add_prefix('feature_')

        f_myfile = open(config["MAGNETO"]["autoencoderPath"] + 'Xtrain_auto.pickle', 'wb')
        pickle.dump(data["Xtrain"], f_myfile)
        f_myfile.close()

        f_myfile = open(config["MAGNETO"]["autoencoderPath"] + 'Xtest_auto.pickle', 'wb')
        pickle.dump(data["Xtest"], f_myfile)
        f_myfile.close()

    if json.loads(dataset_param["featureToDelete"]):
        for i in json.loads(dataset_param["featureToDelete"]):
            del data["Xtrain"][i]
            del data["Xtest"][i]
    data["Xtrain"] = data["Xtrain"].astype(float)
    data["Xtest"] = data["Xtest"].astype(float)

    if toBinary:
        print("[+]Mapping To Binary")
        data["Classification"] = data["Classification"].map(toBinaryMap)
        data["Ytest"] = data["Ytest"].map(toBinaryMap)
    else:
        data["Classification"] = data["Classification"].astype(float)
        data["Ytest"] = data["Ytest"].astype(float)

    f_myfile = open(config["MAGNETO"]["OutputDirMagneto"] + 'Ytrain.pickle', 'wb')
    pickle.dump(data["Classification"], f_myfile)
    f_myfile.close()

    f_myfile = open(config["MAGNETO"]["OutputDirMagneto"] + 'Ytest.pickle', 'wb')
    pickle.dump(data["Ytest"], f_myfile)
    f_myfile.close()

    return VecToImage.toImage(config["MAGNETO"], data, norm=True)
