import tensorflow_addons as tfa
import keras
def model_compile(model,config):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=config.getfloat("MODEL","Lr"), weight_decay=config.getfloat("MODEL","Decay")
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.losses.SparseCategoricalCrossentropy(name="loss"),
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    return model
