from typing import Tuple, List, Dict

import numpy as np
from PIL import Image
from keras import Model
from keras.layers import Conv2D, Dense


def resize_array(arr: np.ndarray, size: Tuple[int, int], rgb: bool):
    """
    Resize an numpy array containing images according to size.
    Assumes following structure [num_images, width, height, [num_channels]]
    :param rgb: whether images should be rgb or grayscale
    :param arr:
    :param size:
    :return: the array with width and height set to the desired structure
    """
    arr = arr.copy()
    has_empty_dimension = arr.shape[-1] == 1
    arr = arr.squeeze()
    new_arr_shape = (arr.shape[0], *size)
    if rgb:
        new_arr_shape = (*new_arr_shape, 3)
    new_arr = np.zeros(new_arr_shape, dtype=arr.dtype)
    for i, img in enumerate(arr):
        p_img = Image.fromarray(img.squeeze())
        if not rgb:
            p_img = p_img.convert('L')
        else:
            p_img = p_img.convert('RGB')
        p_img = p_img.resize(size=size, resample=Image.LANCZOS)
        img = np.array(p_img)
        new_arr[i] = img
    if has_empty_dimension:
        np.expand_dims(new_arr, axis=-1)
    return new_arr


def feature_maps_of_layers(model: Model, layer_indices: List[int], data: np.ndarray) -> Dict[
    int, np.ndarray]:
    """
    Given a model, compute the feature maps of the different indices
    :param model: the model
    :param layer_indices: the layer indices of the layers from which to compute the feature maps from, layers matching the indices have to be Conv2D layers
    :param data: the test data the model is run on
    :return: dict, keys are layer indices, values are feature maps
    """
    activations_for_layers = {}
    for layer_idx in layer_indices:
        output_layer = model.layers[layer_idx]
        assert isinstance(output_layer, (Conv2D, Dense))
        c_model = Model(inputs=model.inputs, outputs=output_layer.output)
        activation = c_model.predict(data)
        activations_for_layers[layer_idx] = activation
    return activations_for_layers
