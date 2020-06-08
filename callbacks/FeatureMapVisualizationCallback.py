import math
import os
import time
from asyncio import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Sequence, Callable, Any

import numpy as np
from PIL import Image
from keras import Model
from keras.callbacks import Callback
from keras.layers import Conv2D
from matplotlib import pyplot as plt

from utils.future_handling import check_finished_futures_and_return_unfinished


class FeatureMapVisualizationCallback(Callback):
    def __init__(self, log_dir: str, model: Model, model_name: str,
                 print_every_n_batches: int, x_train: np.ndarray, num_samples: int = 5,
                 x_train_transform: Callable[[np.ndarray, Any], np.ndarray] = None,
                 transform_params: Sequence[Any] = None, executor: ThreadPoolExecutor = None):
        super().__init__()
        self.model_name = model_name
        self.transform_params = transform_params
        self.x_train_transform = x_train_transform
        self.model = model
        self.log_dir = log_dir
        self.print_every_n_batches = print_every_n_batches
        self.seen = 0
        self.x_train = x_train
        self.fmas = {}
        self.batch_nrs = []
        self.threadpool = ThreadPoolExecutor(max_workers=2) if executor is None else executor
        self.epoch = 1
        self.futures: Sequence[Future] = []
        self.num_samples = num_samples
        assert self.num_samples > 0
        self.sample_idxs = np.random.randint(0, len(self.x_train), self.num_samples)

        self._output_layers = [l for l in self.model.layers if
                               isinstance(l, Conv2D)]
        self._multi_output_model = Model(model.inputs, [l.output for l in self._output_layers])

    @staticmethod
    def _save_feature_maps(img_path_layer, fms, fig_num):
        feature_maps = np.copy(fms[0])
        feature_maps = np.moveaxis(feature_maps, -1, 0)

        min_value = fms.min()
        max_value = fms.max()

        rows = int(math.floor(math.sqrt(len(feature_maps))))
        cols = int(math.ceil(len(feature_maps) / rows))

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), num=fig_num)
        if cols == 1:
            axs = [axs]
        if rows == 1:
            axs = [axs]

        for row in range(rows):
            for col in range(cols):
                fig_idx = row * cols + col
                axs[row][col].set_xticks([])
                axs[row][col].set_yticks([])
                if fig_idx >= len(feature_maps):
                    break
                im = axs[row][col].imshow(feature_maps[fig_idx].squeeze(), vmin=min_value,
                                          vmax=max_value)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig("{}.png".format(img_path_layer))
        fig.clear()
        plt.close(fig)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.futures = check_finished_futures_and_return_unfinished(self.futures)
        if batch % self.print_every_n_batches == 0 and len(self._output_layers) > 0:
            samples = np.copy(self.x_train[self.sample_idxs])
            x_train = samples if self.x_train_transform is None else self.x_train_transform(samples,
                                                                                            *self.transform_params)

            self.seen += 1
            self.batch_nrs.append(batch)
            for sample_nr, sample in enumerate(
                    x_train if self.x_train_transform is None else np.array(x_train).swapaxes(1, 0)):

                # draw sample from data
                sample_as_uint8 = np.copy(samples[sample_nr])

                if not sample_as_uint8.dtype == np.uint8:
                    sample_as_uint8 *= 255.0
                    sample_as_uint8 = sample_as_uint8.astype(np.uint8)
                sample_as_uint8 = sample_as_uint8.squeeze()
                img_path = os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen),
                                        "feature_map", self.model_name,
                                        "sample_{}".format(sample_nr))
                os.makedirs(img_path, exist_ok=True)
                Image.fromarray(sample_as_uint8).save(
                    os.path.join(img_path, "original.jpg"))

                outputs = self._multi_output_model.predict(
                    np.expand_dims(sample, 0) if self.x_train_transform is None else [np.expand_dims(s, 0) for s in
                                                                                      sample])

                for i, output in enumerate(outputs):
                    img_path_layer = os.path.join(img_path, "{}".format(self._output_layers[i].name))

                    # as the printing a very long time, we do this in threads
                    f = self.threadpool.submit(self._save_feature_maps, *(img_path_layer, output,
                                                                          round(time.time() * 10E6)))
                    self.futures.append(f)
                    fmas = np.sum(np.abs(output), axis=tuple(range(len(output.shape)))[:-1])
                    fmas = fmas
                    self.fmas.setdefault(sample_nr, {})
                    self.fmas[sample_nr].setdefault(self._output_layers[i].name, [])
                    self.fmas[sample_nr][self._output_layers[i].name].append(fmas)
                    fmas_stacked = np.copy(np.stack(self.fmas[sample_nr][self._output_layers[i].name]))
                    fmas_stacked /= np.max(fmas_stacked)

    def on_train_end(self, logs=None):
        self.threadpool.shutdown()
        for f in self.futures:
            f.result()
