import logging
import os
from typing import Union

import numpy as np
from PIL import Image
from keras.callbacks import Callback


class LossLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        logging.info("Begin of epoch {}".format(epoch + 1))
        self.epoch = epoch + 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        string = []
        for k, v in logs.items():
            if isinstance(v, (np.float32, np.float64)):
                string.append("{}: {:.4f}".format(k, v))
        string = ", ".join(string)
        logging.info("epoch {}, batch {}: {}".format(self.epoch, batch + 1, string))
