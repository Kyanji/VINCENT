from keras import Model
from keras.layers import *
from tensorflow import expand_dims
from tensorflow import reduce_mean
from visual_attention import PixelAttention2D


def teacher_CNN( input_shape, num_classes, hyperparameters):
    inputs = Input(input_shape)
    X = Conv2D(32, (hyperparameters["kernel"], hyperparameters["kernel"]), activation='relu', name='conv0',
               kernel_initializer='glorot_uniform')(inputs)
    X = Dropout(rate=hyperparameters['dropout1'])(X)
    X = Conv2D(64, (hyperparameters["kernel"], hyperparameters["kernel"]), activation='relu', name='conv1',
               kernel_initializer='glorot_uniform')(X)
    X = Dropout(rate=hyperparameters['dropout2'])(X)
    X = Conv2D(128, (hyperparameters["kernel"], hyperparameters["kernel"]), activation='relu', name='conv2',
               kernel_initializer='glorot_uniform')(X)
    X = PixelAttention2D(X.shape[-1])(X)
    X = Lambda(lambda x: expand_dims(reduce_mean(x, axis=-1), -1),name="LAMBDA")(X)

    X = Flatten()(X)
    X = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(X)
    X = Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(X)
    X = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(X)
    model = Model(inputs, X)
    return model
# feature_map_model = Model(inputs=model.input, outputs=model.get_layer("lambda").output)
