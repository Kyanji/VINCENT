import keras


def fit(model, config, x_train, y_train, x_val, y_val, callbacks=[]):
    if config.getboolean("MODEL", "EarlyStop"):
        stop_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001,
                                                      patience=config.getint("MODEL", "Patience"),
                                                      restore_best_weights=True, verbose=2)
        callbacks.append(stop_callback)

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config.getint("MODEL", "BatchSize"),
        epochs=config.getint("MODEL", "Epochs"),
        validation_data=(x_val, y_val),
        verbose=1,
        callbacks=callbacks,
    )
    return model, history
