import tensorflow as tf
from keras.utils import to_categorical


def fit(model, config, x_train, y_train, x_val, y_val, callbacks=[]):
    # tf.compat.v1.set_random_seed(config.getint("SETTINGS","Seed"))
    tf.keras.utils.set_random_seed(config.getint("SETTINGS", "Seed"))
    tf.random.set_seed(config.getint("SETTINGS", "Seed"))  # tensorflow 2.x

    if config.getboolean("MODEL", "EarlyStop"):
        stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                                         patience=config.getint("MODEL", "Patience"),
                                                         restore_best_weights=True, verbose=2)
        callbacks.append(stop_callback)

    one_hot_encode_train = to_categorical(y_train)
    one_hot_encode_val = to_categorical(y_val)

    history = model.fit(
        x=x_train,
        y=one_hot_encode_train,
        batch_size=config.getint("MODEL", "BatchSize"),
        epochs=config.getint("MODEL", "Epochs"),
        validation_data=(x_val, one_hot_encode_val),
        verbose=1,
        callbacks=callbacks,
    )
    return model, history
