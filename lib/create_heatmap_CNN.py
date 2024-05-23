import cv2
import numpy as np
import tensorflow as tf
from baseline_cnn import split
from tensorflow.keras import Model


def createMapAttention(model, X, norm=True):
    weights = Model(inputs=model.input, outputs=model.get_layer("LAMBDA").output)
    out = weights.predict(X)
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    new_out = []
    for e, m in enumerate(out):
        new_out.append(cv2.resize(out[e], (X.shape[2], X.shape[1]), interpolation=cv2.INTER_NEAREST)[..., np.newaxis])
    new_out = np.array(new_out)
    new_out = (new_out - new_out.min(axis=0)) / (new_out.max(axis=0) - new_out.min(axis=0))
    return new_out * X, new_out


#  mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[
#       ..., np.newaxis
#   ]
#  return (mask * image).astype("uint8"), mask


def create_heatmap_CNN(teacher, data, split_rate):
    im = []
    nat = []
    index = list(split(range(len(data)), split_rate))
    for k in index:
        im1, native = createMapAttention(teacher, data[k])
        im.extend(im1)
        nat.extend(native)
    im = np.array(im)
    nat = np.array(nat)
    return im, nat
