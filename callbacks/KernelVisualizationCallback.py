import logging
import math
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from keras import Model
from keras.callbacks import Callback
from keras.layers import Conv2D
from matplotlib import pyplot as plt

from utils.future_handling import check_finished_futures_and_return_unfinished


class KernelVisualizationCallback(Callback):
    def __init__(self, log_dir: str, print_every_n_batches: int, model: Model, model_name: str,
                 executor: ThreadPoolExecutor = None):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.log_dir = log_dir
        self.seen = 0
        self.epoch = 1
        self.threadpool = ThreadPoolExecutor(max_workers=2) if executor is None else executor
        self.print_every_n_batches = print_every_n_batches
        self._conv_layers = [l for l in self.model.layers if
                             isinstance(l, Conv2D)]
        self.futures = []

    @staticmethod
    def _plot_kernels(filters, img_path, layer_name, fig_num):

        rows = int(math.floor(math.sqrt(len(filters))))
        cols = int(math.ceil(len(filters) / rows))

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), num=fig_num)

        if cols == 1:
            axs = [axs]
        if rows == 1:
            axs = [axs]

        min, max = np.min(filters), np.max(filters)
        for row in range(rows):
            for col in range(cols):
                fig_idx = row * cols + col
                axs[row][col].set_xticks([])
                axs[row][col].set_yticks([])
                if fig_idx >= len(filters):
                    break
                im = axs[row][col].imshow(filters[fig_idx].squeeze(), vmin=min, vmax=max)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(os.path.join(img_path, "{}.png".format(layer_name)))
        fig.clear()
        plt.close(fig)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.futures = check_finished_futures_and_return_unfinished(self.futures)
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            logging.info("Visualizing kernels")
            # summarize filter shapes
            # get filter weights
            # retrieve weights from the second hidden layer
            for layer in self._conv_layers:
                filters = layer.get_weights()[0]
                # normalize filter values to 0-1 so we can visualize them
                filters = np.moveaxis(filters, (0, 1), (-2, -1))
                filters = (filters.reshape(((-1,) + filters.shape[-2:])))
                img_path = os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen),
                                        "kernels", self.model_name)
                os.makedirs(img_path, exist_ok=True)
                f = self.threadpool.submit(self._plot_kernels,
                                           *(filters, img_path, layer.name, round(time.time() * 10E6)))
                self.futures.append(f)

    def on_train_end(self, logs=None):
        self.threadpool.shutdown()
        for f in self.futures:
            f.result()
