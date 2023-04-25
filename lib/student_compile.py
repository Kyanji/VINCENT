import tensorflow_addons as tfa
import keras


def student_compile(model, hyperparam):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=hyperparam["learning_rate"],weight_decay=0.01
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
