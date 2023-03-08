from keras_cv_attention_models import botnet


def load_model(config,input_shape,num_classes):
    if config["MODEL"]["Model"]=="Botnet50":
        model = botnet.BotNet50(input_shape=input_shape, pretrained='imagenet', num_classes=num_classes )
    elif config["MODEL"]["Model"]=="Botnet26T":
        model = botnet.BotNet26T(input_shape=input_shape,pretrained='imagenet', num_classes=num_classes,)

    return model