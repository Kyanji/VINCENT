from keras_cv_attention_models import botnet
import tensorflow as tf

from lib.custom_vit import VisualTransformers


def load_model(config, input_shape, num_classes):
    tf.compat.v1.set_random_seed(config.getint("SETTINGS", "Seed"))
    tf.keras.utils.set_random_seed(config.getint("SETTINGS", "Seed"))

    if len(input_shape) == 2:
        chan = 1
    else:
        chan = input_shape[2]
    model = VisualTransformers(input_shape[0], chan, config.getint("VIT_SETTINGS", "PatchSize"),
                               config.getint("VIT_SETTINGS", "NumLayer"),
                               config.getint("VIT_SETTINGS", "HiddenDim"),
                               config.getint("VIT_SETTINGS", "NumHeads"), config.getint("VIT_SETTINGS", "MlpDim"),
                               num_classes, config.getfloat("VIT_SETTINGS", "Dropout"))
    return model
