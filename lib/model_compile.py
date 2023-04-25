import tensorflow_addons as tfa
import keras


def model_compile(model, config):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=config.getfloat("MODEL", "Lr"), weight_decay=config.getfloat("MODEL", "Decay")
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.losses.CategoricalCrossentropy(name="loss"),
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )

    return model
