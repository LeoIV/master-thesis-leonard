import logging
import os
import time
from concurrent import futures
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.future_handling import check_finished_futures_and_return_unfinished


class KernelVisualizationCallback(Callback):
    def __init__(self, log_dir: str, print_every_n_batches: int,
                 layer_idx: int):
        super().__init__()
        self.log_dir = log_dir
        self.seen = 0
        self.epoch = 1
        self.threadpool = ThreadPoolExecutor(max_workers=5)
        self.print_every_n_batches = print_every_n_batches
        self.layer_idx = layer_idx
        self.futures = []

    @staticmethod
    def _plot_kernels(filters, img_path, fig_num):
        fig = plt.figure(fig_num, figsize=((8.0, 6.0)))
        for map_nr, f_map in enumerate(filters):
            ax = fig.gca()
            img = ax.imshow(f_map.squeeze())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax)
            fig.savefig(os.path.join(img_path, "map_{}.png".format(map_nr)))
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
            filters = self.model.layers[self.layer_idx].get_weights()[0]
            # normalize filter values to 0-1 so we can visualize them
            filters = np.moveaxis(filters, (0, 1), (-2, -1))
            filters = (filters.reshape(((-1,) + filters.shape[-2:])))
            img_path = os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen),
                                    "layer1_kernels")
            os.makedirs(img_path, exist_ok=True)
            f = self.threadpool.submit(self._plot_kernels, *(filters, img_path, round(time.time() * 10E6)))
            self.futures.append(f)

    def on_train_end(self, logs=None):
        self.threadpool.shutdown()
        for f in self.futures:
            f.result()
