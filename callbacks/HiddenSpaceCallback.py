import logging
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional

import numpy as np
from keras import Model
from keras.callbacks import Callback
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from utils.future_handling import check_finished_futures_and_return_unfinished


class HiddenSpaceCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VAEWrapper', batch_size: int, x_train: np.ndarray,
                 y_train: Optional[np.ndarray] = None, max_samples: int = 1000):
        """
        Visualize the embedding space using T-SNE
        :param log_dir:
        :param vae:
        :param x_train: the data of the embedded space to visualize
        :param y_train: optionally labels to color the points according to their classes
        :param max_samples:
        """
        super().__init__()
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

        encoder = self.vae.encoder
        encoder_input = encoder.inputs
        mu_layer = [l for l in encoder.layers if l.name == "mu"]
        if not len(mu_layer) == 1:
            raise RuntimeError("the HiddenSpaceCallback requires the encoder to have exactly one layer called mu")
        mu_layer = mu_layer[0]
        self.encoder_til_mu = Model(encoder_input, mu_layer.output)

    @staticmethod
    def _print_embeddings(mus: np.ndarray, y: Optional[np.ndarray], fig_path: str):
        mus_embedded = TSNE().fit_transform(mus)
        fig, ax = plt.subplots(num=round(time.time() * 10E6), figsize=(8, 6))

        ax.scatter(mus_embedded[:, 0], mus_embedded[:, 1], c=y, alpha=0.7)
        ax.legend()
        fig.savefig(os.path.join(fig_path, 'embeddings.png'))
        plt.close(fig)

    def on_batch_end(self, batch, logs=None):
        self.futures = check_finished_futures_and_return_unfinished(self.futures)

    def on_epoch_end(self, epoch, logs=None):
        logging.info("Visualizing hidden space")
        mus = self.encoder_til_mu.predict(self.x_train)
        fig_path = os.path.join(self.log_dir, "epoch_{}".format(epoch + 1))
        os.makedirs(fig_path, exist_ok=True)
        f = self.threadpool.submit(self._print_embeddings, *(mus, self.y_train, fig_path))
        self.futures.append(f)

    def on_train_end(self, logs=None):
        self.threadpool.shutdown()
        for f in self.futures:
            f.result()
