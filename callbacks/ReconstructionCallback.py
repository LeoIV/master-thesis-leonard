import logging
import os

from PIL import Image
from keras.callbacks import Callback
import numpy as np


class ReconstructionImagesCallback(Callback):
    """
    Randomly draw 9 Gaussians and reconstruct their images with fixed seeds (consistency over batches)
    """

    def __init__(self, log_dir: str, print_every_n_batches: int, initial_epoch: int, vae: 'VariationalAutoencoder',
                 x_train: np.ndarray, num_reconstructions: int = 10, num_inputs: int = 1):
        """

        :type num_inputs: vlae has multiple z_dim inputs, other vaes one
        :param log_dir:
        :param print_every_n_batches:
        :param initial_epoch:
        :param vae:
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.x_train = x_train
        self.epoch = initial_epoch
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
        self.seeds = list(range(num_reconstructions))
        self.seen = 0
        self.epoch = 1
        self.log_dir = log_dir
        idxs = np.random.randint(0, len(self.x_train), num_reconstructions)
        self.samples = [np.copy(self.x_train[idx]) for idx in idxs]

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            logging.info("Visualizing reconstructions")
            img_path = os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen),
                                    "generated_reconstructed")
            os.makedirs(img_path, exist_ok=True)
            for seed in self.seeds:
                # make sure we always reconstruct the same image
                np.random.seed(seed)
                z_news = [np.random.normal(size=(1, self.vae.z_dims[i])) for i in range(self.num_inputs)]
                np.random.seed(None)
                # predictions are in [0-1] float format
                gen = self.vae.decoder.predict(z_news if len(z_news) > 1 else z_news[0])
                # make byte array
                k = (gen.squeeze() * 255.0).astype(np.uint8)
                Image.fromarray(k.squeeze()).save(os.path.join(img_path, "generated_{}.jpg".format(seed)))
            for i, sample in enumerate(self.samples):
                Image.fromarray((sample * 255.0).astype(np.uint8)).save(
                    os.path.join(img_path, "original_{}.jpg".format(i)))
                reconst = self.vae.model.predict(np.expand_dims(sample, 0))
                k = (reconst.squeeze() * 255.0).astype(np.uint8)
                Image.fromarray(k.squeeze()).save(os.path.join(img_path, "reconstructed_{}.jpg".format(i)))
