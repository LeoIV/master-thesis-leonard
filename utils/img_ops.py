from typing import Tuple

import numpy as np
from PIL import Image


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
