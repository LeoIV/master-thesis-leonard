import math
import time
from typing import Tuple, List, Dict, Union, Iterable

import numpy as np
from PIL import Image
from keras import Model
from keras.layers import Conv2D, Dense
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


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


def filters_to_figure(filters: Union[np.array, Iterable[np.array]], filter_spacing: int = 2, c_map: str = 'viridis',
                      fig_num: int = None) -> Figure:
    """
    Instead of plotting each filter ('filter' = kernel or feature map) in a single subplot, create a new numpy array
    from the filters with appropriate spacing and plot this in a single subplot (galaxies faster)
    :param filters: the filters, has to be an Iterable of 2D arrays (arrays have to be squared and have to have the same shape)
    :param filter_spacing: the spacing between the filters in the plot (default 2 pixels)
    :param c_map: the colormap to use to color the filters (a matplotlib supported colormap)
    :param fig_num: the number of the figure
    :return: a Figure that then for example can be saved as an image
    """
    # we need all filters to have the same size and to be square so the filter size (in one dimension) is:
    filter_size = filters[0].shape[0]
    # ensure constraints
    for flt in filters:
        assert flt.shape[0] == flt.shape[1] == filter_size
        assert len(flt.shape) == 2

    c_map = cm.get_cmap(c_map)
    # We want the space between the filters to be transparent as fig.save_fig also adds transparent space everywhere
    # else. We achieve this by setting the array values to a value lower than the minimum of the feature maps
    # and by then letting the feature map everything below the minimum of the feature maps to a transparent RGBA
    # value
    c_map.set_under('w', alpha=1.0)

    # make figure approximately square
    rows = int(math.floor(math.sqrt(len(filters))))
    cols = int(math.ceil(len(filters) / rows))

    # create figure with one subplot
    fig, ax = plt.subplots(num=round(time.time() * 10E6) if fig_num is None else fig_num,
                           figsize=(15, 13))

    if not isinstance(filters, np.ndarray):
        filters = np.ndarray(filters)

    min, max = np.min(filters), np.max(filters)

    array_cols = filter_size * cols + (cols - 1) * filter_spacing
    array_rows = filter_size * rows + (rows - 1) * filter_spacing

    # create array with default value minimum of the feature maps - 1
    array = np.full(shape=(array_rows, array_cols), fill_value=min - 1, dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            filter_idx = row * cols + col
            if filter_idx >= len(filters):
                break
            # row and column in the numpy array
            arr_row_idx = row * (filter_size + filter_spacing)
            arr_col_idx = col * (filter_size + filter_spacing)
            # fill array at the position with the feature map
            array[arr_row_idx:arr_row_idx + filter_size, arr_col_idx:arr_col_idx + filter_size] = filters[filter_idx]
    # initialize normalizer with minimum and maximum obtained above
    norm = Normalize(vmin=min - 1E-3, vmax=max + 1E-3)
    # remove ticks and border around subplot
    ax.axis('off')
    # imshow with normalizer and custom colormap
    im = ax.imshow(array, cmap=c_map, norm=norm)
    # add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig
