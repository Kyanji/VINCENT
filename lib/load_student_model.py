from keras import Model
from keras.layers import *


def load_student(config, input_shape, num_classes, hyperparameters, test_time=False):
    if test_time:
        hyperparameters["dropout1"] = 0
        hyperparameters["dropout2"] = 0
    if config["DISTILLATION"]["model"] == "CNN1":
        inputs = Input(input_shape)
        X = Conv2D(32, (hyperparameters["kernel"], hyperparameters["kernel"]), activation='relu', name='conv0',
                   kernel_initializer='glorot_uniform')(inputs)
        X = Dropout(rate=hyperparameters['dropout1'])(X)
        X = Conv2D(64, (hyperparameters["kernel"], hyperparameters["kernel"]), activation='relu', name='conv1',
                   kernel_initializer='glorot_uniform')(X)
        X = Dropout(rate=hyperparameters['dropout2'])(X)
        X = Conv2D(128, (hyperparameters["kernel"], hyperparameters["kernel"]), activation='relu', name='conv2',
                   kernel_initializer='glorot_uniform')(X)
        X = Flatten()(X)
        X = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(X)
        X = Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(X)
        X = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(X)
        model = Model(inputs, X)

    return model
