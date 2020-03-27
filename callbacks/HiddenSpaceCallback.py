import logging
import os
import time
from collections import Iterable
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Sequence, Dict, List

import numpy as np
from keras import Model
from keras.callbacks import Callback
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from utils.future_handling import check_finished_futures_and_return_unfinished


class HiddenSpaceCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VAEWrapper', batch_size: int, x_train: np.ndarray,
                 layer_names: Sequence[str], y_train: Optional[np.ndarray] = None, max_samples: int = 1000,
                 plot_params: List[Dict[str, any]] = None):
        """
        Visualize the embedding space using T-SNE
        :param log_dir:
        :param vae:
        :param x_train: the data of the embedded space to visualize
        :param y_train: optionally labels to color the points according to their classes
        :param max_samples:
        """
        super().__init__()
        if plot_params is None:
            plot_params = []
        self.plot_params = plot_params
        self.batch_size = batch_size
        self.y_train = y_train
        self.x_train = x_train
        self.max_samples = max_samples
        self.vae = vae
        self.futures = []
        self.log_dir = log_dir
        self.threadpool = ThreadPoolExecutor(max_workers=2)
        if len(self.x_train) > self.max_samples:
            idxs = np.random.randint(0, len(self.x_train), self.max_samples)
            self.x_train = np.array([np.copy(self.x_train[idx]) for idx in idxs])
            if self.y_train:
                self.y_train = np.array([np.copy(self.y_train[idx]) for idx in idxs])
            for k, v in self.plot_params.items():
                self.plot_params[k] = np.array([np.copy(v[idx]) for idx in idxs])
        else:
            num_samples = len(self.x_train)
            for i, _ in enumerate(self.plot_params):
                for k, v in self.plot_params[i].items():
                    if isinstance(v, np.ndarray):
                        self.plot_params[i][k] = np.copy(v)[:num_samples]

        encoder = self.vae.encoder
        encoder_input = encoder.inputs
        mu_layers = [l for l in encoder.layers if l.name in layer_names]
        if not len(mu_layers) == len(layer_names):
            logging.error("Didn't find all layers for hidden space visualization")
            # raise RuntimeError("Didn't find all layers for hidden space visualization")
        logging.info("Using embedding layers: {}".format([l.name for l in mu_layers]))
        self.encoders_til_mu = [Model(encoder_input, ml.output) for ml in mu_layers]

    @staticmethod
    def _print_embeddings(mus: np.ndarray, y: Optional[np.ndarray], fig_path: str, layer_name: str,
                          plot_params=None):
        if plot_params is None:
            plot_params = {}
        mus_embedded = TSNE().fit_transform(mus) if mus.shape[-1] != 2 else mus
        fig, ax = plt.subplots(num=round(time.time() * 10E6), figsize=(8, 6))

        ax.scatter(mus_embedded[:, 0], mus_embedded[:, 1], alpha=0.5, **plot_params)
        ax.plot()
        ax.legend()
        fig.savefig(os.path.join(fig_path, "embeddings_{}.png".format(layer_name)))
        plt.close(fig)

    def on_batch_end(self, batch, logs=None):
        self.futures = check_finished_futures_and_return_unfinished(self.futures)

    def on_epoch_end(self, epoch, logs=None):
        logging.info("Visualizing hidden space")
        for encoder_til_mu in self.encoders_til_mu:
            mus = encoder_til_mu.predict(self.x_train)
            fig_path = os.path.join(self.log_dir, "epoch_{}".format(epoch + 1), "embeddings")
            os.makedirs(fig_path, exist_ok=True)
            if len(self.plot_params) > 0:
                for i, plot_params in enumerate(self.plot_params):
                    f = self.threadpool.submit(self._print_embeddings,
                                               *(mus, self.y_train, fig_path,
                                                 "{}_{}".format(encoder_til_mu.layers[-1].name, i), plot_params))
                    self.futures.append(f)
            else:
                f = self.threadpool.submit(self._print_embeddings,
                                           *(mus, self.y_train, fig_path,
                                             encoder_til_mu.layers[-1].name))
                self.futures.append(f)

    def on_train_end(self, logs=None):
        self.threadpool.shutdown()
        for f in self.futures:
            f.result()
