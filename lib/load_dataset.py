import pickle

import numpy as np


def load_dataset(dataset_param):
    f_myfile = open(dataset_param["OutputDirMagneto"] + dataset_param["trainName"], 'rb')
    x_train = np.array(pickle.load(f_myfile))
    f_myfile.close()

    f_myfile = open(dataset_param["OutputDirMagneto"] + dataset_param["YtrainName"], 'rb')
    y_train = np.array(pickle.load(f_myfile))
    f_myfile.close()

    f_myfile = open(dataset_param["OutputDirMagneto"] + dataset_param["testName"], 'rb')
    x_test = np.array(pickle.load(f_myfile))
    f_myfile.close()

    f_myfile = open(dataset_param["OutputDirMagneto"] + dataset_param["YtestName"], 'rb')
    y_test =  np.array(pickle.load(f_myfile))
    f_myfile.close()

    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    return x_train, y_train, x_test, y_test
