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
        self.batch_logdir = ".".join(logfile.split(".")[:-1]) + "_batch.csv"
        self.epoch_logdir = ".".join(logfile.split(".")[:-1]) + "_epoch.csv"
        self.logdir = logdir
        self._bf = open(os.path.join(logdir, self.batch_logdir), mode="w", newline='')
        self._ef = open(os.path.join(logdir, self.epoch_logdir), mode="w", newline='')
        self.batch_csv_writer = csv.writer(self._bf, delimiter=",")
        self.epoch_csv_writer = csv.writer(self._ef, delimiter=",")
        self.batch_lines_written = 0
        self.epoch_lines_written = 0
        self.batch_headers = ["time", "batch"]
        self.epoch_headers = ["time", "epoch"]

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
                if self.batch_lines_written == 0:
                    self.batch_headers.append(k)
                losses.append(v)
                string.append("{}: {:.4f}".format(k, v))
        if self.batch_lines_written == 0:
            self.batch_csv_writer.writerow(self.batch_headers)
        self.batch_csv_writer.writerow(losses)
        self.batch_lines_written += 1
        string = ", ".join(string)
        logging.info("epoch {}, batch {}: {}".format(self.epoch, batch + 1, string))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        string = []
        losses = [datetime.now(), epoch]
        for k, v in logs.items():
            if isinstance(v, (np.float32, np.float64)):
                if self.epoch_lines_written == 0:
                    self.epoch_headers.append(k)
                losses.append(v)
                string.append("{}: {:.4f}".format(k, v))
        if self.epoch_lines_written == 0:
            self.epoch_csv_writer.writerow(self.epoch_headers)
        self.epoch_csv_writer.writerow(losses)
        self.epoch_lines_written += 1
        string = ", ".join(string)
        logging.info("epoch {}: {}".format(self.epoch, string))

        self._bf.flush()
        self._ef.flush()
        losses = np.loadtxt(os.path.join(self.logdir, self.batch_logdir), skiprows=1,
                            usecols=list(range(len(self.batch_headers)))[2:], delimiter=",")
        loss_names = self.batch_headers[2:]
        fig = plt.figure(num=round(time.time() * 10E6))
        ax = fig.gca()
        arr = np.arange(len(losses))
        for loss in losses.T:
            ax.plot(arr, loss)
        ax.legend(loss_names, loc='upper right')
        fig.savefig(os.path.join(self.logdir, "batch_losses.png"), mode="w+")

        losses = np.loadtxt(os.path.join(self.logdir, self.epoch_logdir), skiprows=1,
                            usecols=list(range(len(self.epoch_headers)))[2:], delimiter=",")
        loss_names = self.epoch_headers[2:]
        fig = plt.figure(num=round(time.time() * 10E6))
        ax = fig.gca()
        losses = losses[np.newaxis, :] if len(losses.shape) == 1 else losses
        arr = np.arange(len(losses))
        for loss in losses.T:
            ax.plot(arr, loss)
        ax.legend(loss_names, loc='upper right')
        fig.savefig(os.path.join(self.logdir, "epoch_losses.png"), mode="w+")

    def on_train_end(self, logs=None):
        self._bf.close()
