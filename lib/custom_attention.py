import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from vit_keras import layers, vit


def attention_map_no_norm(model, image):
    """Get an attention map for an image and model using the technique
    described in Appendix D.7 in the paper (unofficial).

    Args:
        model: A ViT model
        image: An image for which we will compute the attention map.
    """
    size = model.input_shape[1]
    grid_size = int(np.sqrt(model.layers[5].output_shape[0][-2] - 1))

    # Prepare the input
    X = image.reshape(1, size, size, 3)  # type: ignore

    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if isinstance(l, layers.TransformerBlock)
    ]
    weights = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    )
    num_layers = weights.shape[0]
    num_heads = weights.shape[2]
    reshaped = weights.reshape(
        (num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[
        ..., np.newaxis
    ]
    return (mask * image).astype("uint8"), mask


def attention_map_no_norm_torch(model, image,scale=False):
    """Get an attention map for an image and model using the technique
    described in Appendix D.7 in the paper (unofficial).

    Args:
        model: A ViT model
        image: An image for which we will compute the attention map.
    """
    # size = model.input_shape[1]
    grid_size = int(np.sqrt(model.layers[5].output_shape[0][-2] - 1))

    # Prepare the input
    image = tf.expand_dims(image, axis=0)
    X = image
    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if isinstance(l, layers.TransformerBlock)
    ]
    weights = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs)(X)
    )
    num_layers = weights.shape[0]
    num_heads = weights.shape[2]
    reshaped = weights.reshape(
        (num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[
        ..., np.newaxis
    ]
    return (mask * image.numpy()).astype("uint8"), mask


def Swap(arr, start_index, last_index):
    arr[:, [start_index, last_index]] = arr[:, [last_index, start_index]]


def attention_map_no_norm_fast(model, image,scale=False):
    """Get an attention map for an image and model using the technique
    described in Appendix D.7 in the paper (unofficial).

    Args:
        model: A ViT model
        image: An image for which we will compute the attention map.
    """
    size = model.input_shape[1]
    grid_size = int(np.sqrt(model.layers[5].output_shape[0][-2] - 1))

    # Prepare the input
    X = image.reshape(-1, size, size, 3)  # type: ignore

    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if isinstance(l, layers.TransformerBlock)
    ]
    weights = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    )
    num_layers = weights.shape[0]
    num_heads = weights.shape[2]
    w = np.swapaxes(weights, 0, 1)
    reshaped = w.reshape(
        (-1, num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    mean = np.array([r.mean(axis=1) for r in reshaped])
    # reshaped = reshaped.mean(axis=1)

    # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = [r + np.eye(r.shape[1]) for r in mean]

    # reshaped = reshaped + np.eye(reshaped.shape[1])

    reshaped = np.array([r / r.sum(axis=(1, 2))[:, np.newaxis, np.newaxis] for r in reshaped])

    # reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]
    # Recursively multiply the weight matrices
    #
    v1 = []
    for i in reshaped:
        v = i[-1]
        for n in range(1, len(i)):
            v = np.matmul(v, i[-1 - n])
        v1.append(v)
    v1 = np.array(v1)
    # Attention from the output token to the input space.
    mask = v1[:,0, 1:].reshape(-1,grid_size, grid_size)
    final_mask = np.ones((len(mask),image.shape[1], image.shape[2],1))
    if scale==True:
        for e,m in enumerate( mask):
            final_mask[e]=cv2.resize(m / m.max(), (image.shape[1], image.shape[2]))[
            ..., np.newaxis]
    else:
        for e,m in enumerate( mask):
            final_mask[e]=cv2.resize(m / m.max(), (image.shape[1], image.shape[2]),interpolation=cv2.INTER_NEAREST)[
            ..., np.newaxis]
    X=final_mask
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (1 - 0) + 0

    return (final_mask * image).astype("uint8"), final_mask, (X_scaled * image).astype("uint8")
