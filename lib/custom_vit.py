import tensorflow as tf
import vit_keras
from vit_keras import vit
def VisualTransformers(image_size, num_channels, PatchSize, NumLayer, HiddenDim, NumHeads, MlpDim, NumClasses,Dropout):
    """ Input """
    inputs = tf.keras.layers.Input((image_size, image_size, num_channels)) ## (None, 512, 512, 3)

    patch_embed = tf.keras.layers.Conv2D(
        filters=HiddenDim,
        kernel_size=PatchSize,
        strides=PatchSize,
        padding="valid",
        name="embedding",
    )(inputs)

    """ Patch Embeddings """
    patch_embed = tf.keras.layers.Reshape((patch_embed.shape[1] * patch_embed.shape[2], HiddenDim))(patch_embed)

    """ Position Embeddings """
    x = vit_keras.vit.layers.ClassToken(name="class_token")(patch_embed)
    x = vit_keras.vit.layers.AddPositionEmbs(name="Transformer/posembed_input")(x)

    """ Transformer Encoder """
    from vit_keras import vit
    for n in range(NumLayer):
        x, _ = vit_keras.vit.layers.TransformerBlock(
            num_heads=NumHeads,
            mlp_dim=MlpDim,
            dropout=Dropout,
            name=f"Transformer/encoderblock_{n}",
        )(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="Transformer/encoder_norm")(x)
    x = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)
    x = tf.keras.layers.Dense(NumClasses, name="head", activation='softmax')(x)
    model = tf.keras.models.Model(inputs, x)
    return model
