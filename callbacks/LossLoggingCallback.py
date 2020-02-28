import csv
import logging
import os
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import Callback


class LossLoggingCallback(Callback):
    def __init__(self, logdir: str, logfile: str = "losses.csv"):
        super().__init__()
        self.epoch = 0
        self.logdir = logdir
        self.logfile = logfile
        self._f = open(os.path.join(logdir, logfile), mode="w", newline='')
        self.csv_writer = csv.writer(self._f, delimiter=",")
        self.lines_written = 0
        self.headers = ["time", "batch"]

    def on_epoch_begin(self, epoch, logs=None):
        logging.info("Begin of epoch {}".format(epoch + 1))
        self.epoch = epoch + 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        string = []
        losses = [datetime.now(), batch]
        for k, v in logs.items():
            if isinstance(v, (np.float32, np.float64)):
                if self.lines_written == 0:
                    self.headers.append(k)
                losses.append(v)
                string.append("{}: {:.4f}".format(k, v))
        if self.lines_written == 0:
            self.csv_writer.writerow(self.headers)
        self.csv_writer.writerow(losses)
        self.lines_written += 1
        string = ", ".join(string)
        logging.info("epoch {}, batch {}: {}".format(self.epoch, batch + 1, string))

    def on_epoch_end(self, epoch, logs=None):
        self._f.flush()
        losses = np.loadtxt(os.path.join(self.logdir, self.logfile), skiprows=1,
                            usecols=list(range(len(self.headers)))[2:], delimiter=",")
        loss_names = self.headers[2:]
        fig = plt.figure(num=round(time.time() * 10E6))
        ax = fig.gca()
        arr = np.arange(len(losses))
        for loss in losses.T:
            ax.plot(arr, loss)
        ax.legend(loss_names, loc='upper right')
        fig.savefig(os.path.join(self.logdir, "losses.png"), mode="w+")

    def on_train_end(self, logs=None):
        self._f.close()
