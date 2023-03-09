from keras_cv_attention_models import botnet
import tensorflow as tf

from lib.custom_vit import VisualTransformers


def load_model(config,input_shape,num_classes):
    tf.compat.v1.set_random_seed(config.getint("SETTINGS","Seed"))
    tf.keras.utils.set_random_seed(config.getint("SETTINGS","Seed"))

    if config["MODEL"]["Model"] == "Botnet50":
        model = botnet.BotNet50(input_shape=input_shape, pretrained='imagenet', num_classes=num_classes )
    elif config["MODEL"]["Model"] == "Botnet26T":
        model = botnet.BotNet26T(input_shape=input_shape,pretrained='imagenet', num_classes=num_classes,)
    elif config["MODEL"]["Model"]=="CustomVit":
        model=VisualTransformers(input_shape[0],3, config.getint("VIT_SETTINGS","PatchSize"),config.getint("VIT_SETTINGS","NumLayer"),config.getint("VIT_SETTINGS","HiddenDim"),config.getint("VIT_SETTINGS","NumHeads"),config.getint("VIT_SETTINGS","MlpDim"),num_classes )
    return model