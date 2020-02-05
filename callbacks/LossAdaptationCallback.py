import logging
import os
from typing import Union

import numpy as np
from PIL import Image
from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import LearningRateScheduler


class LossAdaptationCallback(Callback):
    def __init__(self, vae: Union['VariationalAutoencoder', 'AlexNetVAE'], print_every_n_batches: int,
                 update_by: float = 0.01):
        super().__init__()
        self.vae = vae
        self.seen = 0
        self.print_every_n_batches = print_every_n_batches
        self.update_by = update_by
        self.required_steps = 1. / update_by

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1

            if self.seen <= self.required_steps:
                K.set_value(self.vae.kl_loss_factor, self.seen * self.update_by)
                print("increasing kl loss factor {}, currently at {}".format(self.vae.kl_loss_factor,
                                                                             K.eval(self.vae.kl_loss_factor)))
