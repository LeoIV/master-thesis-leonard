import logging
import os
from typing import Union

import numpy as np
from PIL import Image
from keras.callbacks import Callback


class KernelVisualizationCallback(Callback):
    def __init__(self, log_dir: str, vae: Union['VariationalAutoencoder', 'AlexNet'], print_every_n_batches: int,
                 layer_idx: int):
        super().__init__()
        self.vae = vae
        self.log_dir = log_dir
        self.seen = 0
        self.epoch = 1
        self.print_every_n_batches = print_every_n_batches
        self.layer_idx = layer_idx

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            logging.info("Visualizing kernels")
            # summarize filter shapes
            # get filter weights
            # retrieve weights from the second hidden layer
            filters, biases = self.model.layers[self.layer_idx].get_weights()
            # normalize filter values to 0-1 so we can visualize them
            filters = np.moveaxis(filters, (0, 1), (-2, -1))
            filters = (filters.reshape(((-1,) + filters.shape[-2:])))
            img_path = os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen),
                                    "layer1_kernels")
            os.makedirs(img_path, exist_ok=True)
            for map_nr, f_map in enumerate(filters):
                f_min, f_max = f_map.min(), f_map.max()
                f_map = (f_map - f_min) / (f_max - f_min)
                f_map = (f_map * 255.0).astype(np.uint8)
                Image.fromarray(f_map.squeeze()).save(os.path.join(img_path, "map_{}.jpg".format(map_nr)))
