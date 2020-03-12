import logging
import math
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, Sequence

import numpy as np
from PIL import Image
from keras import Model
from keras.callbacks import Callback
from matplotlib import pyplot as plt

from utils.future_handling import check_finished_futures_and_return_unfinished


class FeatureMapVisualizationCallback(Callback):
    def __init__(self, log_dir: str, model_wrapper: Union['VariationalAutoencoder', 'AlexNet'],
                 print_every_n_batches: int,
                 layer_idxs: Sequence[int], x_train: np.ndarray, num_samples: int = 5):
        super().__init__()
        self.log_dir = log_dir
        self.model_wrapper = model_wrapper
        self.print_every_n_batches = print_every_n_batches
        self.layer_idxs = layer_idxs
        self.seen = 0
        self.x_train = x_train
        self.fmas = {}
        self.batch_nrs = []
        self.threadpool = ThreadPoolExecutor(max_workers=5)
        self.epoch = 1
        self.futures = []
        self.num_samples = num_samples
        idxs = np.random.randint(0, len(self.x_train), num_samples)
        self.samples = [np.copy(self.x_train[idx]) for idx in idxs]

    @staticmethod
    def _save_feature_maps(img_path_layer, fms, fig_num):
        feature_maps = np.copy(fms).squeeze()
        feature_maps = np.moveaxis(feature_maps, -1, 0)

        min_value = fms.min()
        max_value = fms.max()

        columns = 10
        rows = math.ceil((len(feature_maps) + 1) / columns)

        fig, axs = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2), num=fig_num)

        for xax in axs:
            if rows > 1:
                for yax in xax:
                    yax.set_xticks([])
                    yax.set_yticks([])
            else:
                xax.set_xticks([])
                xax.set_yticks([])

        for i, f_map in enumerate(feature_maps):
            row = i // columns
            column = i % columns
            (axs[row, column] if rows > 1 else axs[column]).imshow(f_map.squeeze(), vmin=min_value,
                                                                   vmax=max_value)
        fig.colorbar((axs[0, 0] if rows > 1 else axs[0]).get_images()[0], cax=axs[-1, -1] if rows > 1 else axs[-1])
        fig.savefig("{}.png".format(img_path_layer))
        fig.clear()
        plt.close(fig)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.futures = check_finished_futures_and_return_unfinished(self.futures)
        if batch % self.print_every_n_batches == 0:
            plt.rcParams["figure.figsize"] = (7 * self.num_samples, 10 + 10 * (len(self.layer_idxs)))
            self.seen += 1
            self.batch_nrs.append(batch)
            fig, ax = plt.subplots(1 + len(self.layer_idxs), len(self.samples), num=round(time.time() * 10E6))
            for sample_nr, sample in enumerate(self.samples):

                # draw sample from data
                sample_as_uint8 = np.copy(sample)

                if not sample_as_uint8.dtype == np.uint8:
                    sample_as_uint8 *= 255.0
                    sample_as_uint8 = sample_as_uint8.astype(np.uint8)
                sample_as_uint8 = sample_as_uint8.squeeze()
                img_path = os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen),
                                        "feature_map",
                                        "sample_{}".format(sample_nr))
                os.makedirs(img_path, exist_ok=True)
                ax[0, sample_nr].imshow(sample.squeeze(), cmap='gray')
                Image.fromarray(sample_as_uint8).save(
                    os.path.join(img_path, "original.jpg"))
                for i, layer_idx in enumerate(self.layer_idxs):
                    logging.info("Visualizing feature maps for layer {}".format(layer_idx))
                    print("Visualizing feature maps for layer {}".format(layer_idx))

                    # we cannot use instanceof as we aren't allowed to import the model_wrapper class directly since
                    # this would lead to cyclic references
                    if type(self.model_wrapper).__name__ in ['VariationalAutoencoder', 'AlexNetVAE',
                                                             'FrozenAlexNetVAE', 'AlexAlexNetVAE', 'VLAE', 'HVAE']:
                        if layer_idx < len(self.model_wrapper.encoder.layers) - 1:
                            output_layer = self.model_wrapper.encoder.layers[layer_idx]
                            model = Model(inputs=self.model_wrapper.encoder.inputs, outputs=output_layer.output)
                        else:
                            output_layer = self.model_wrapper.decoder.layers[
                                layer_idx - len(self.model_wrapper.encoder.layers) - 1]
                            model = Model(inputs=self.model_wrapper.encoder.inputs,
                                          outputs=Model(self.model_wrapper.decoder.inputs, output_layer.output)(
                                              self.model_wrapper.encoder.outputs))
                    elif type(self.model_wrapper).__name__ in ['AlexNet', 'SimpleClassifier']:
                        output_layer = self.model_wrapper.model.layers[layer_idx]
                        model = Model(inputs=self.model_wrapper.model.inputs, outputs=output_layer.output)
                    else:
                        raise RuntimeError(
                            "model_wrapper has to be either of type VariationalAutoencoder, AlexNetVAE, "
                            "FrozenAlexNetVAE or AlexNet")

                    img_path_layer = os.path.join(img_path, "{}".format(output_layer.name))

                    feature_maps = model.predict(sample[np.newaxis, :], batch_size=1, verbose=1)
                    # as the printing a very long time, we do this in threads
                    f = self.threadpool.submit(self._save_feature_maps, *(img_path_layer, feature_maps,
                                                                          round(time.time() * 10E6)))
                    self.futures.append(f)
                    fmas = np.sum(np.abs(feature_maps), axis=tuple(range(len(feature_maps.shape)))[:-1])
                    fmas = fmas
                    self.fmas.setdefault(sample_nr, {})
                    self.fmas[sample_nr].setdefault(layer_idx, [])
                    self.fmas[sample_nr][layer_idx].append(fmas)
                    fmas_stacked = np.copy(np.stack(self.fmas[sample_nr][layer_idx]))
                    fmas_stacked /= np.max(fmas_stacked)
                    ax[i + 1, sample_nr].set_title('Normalized activations - Layer {}'.format(output_layer.name))
                    ax[i + 1, sample_nr].imshow(fmas_stacked, origin='lower', interpolation="none")
                    ax[i + 1, sample_nr].set(xlabel="#feature map", ylabel="# batch")
                    ax[i + 1, sample_nr].set_yticks(np.arange(len(self.batch_nrs), step=2), self.batch_nrs[0::2])
            fig.colorbar(ax[1, 0].get_images()[0])
            fig.savefig(
                os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen), "feature_map",
                             'activations.png'))
            plt.close(fig)
            # set back to default
            plt.rcParams["figure.figsize"] = (8.0, 6.0)

    def on_train_end(self, logs=None):
        self.threadpool.shutdown()
        for f in self.futures:
            f.result()
