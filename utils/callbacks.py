from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.callbacks import Callback, LearningRateScheduler


class KernelVisualizationCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VariationalAutoencoder', print_every_n_batches: int, initial_epoch: int, filewriter):
        super().__init__()
        self.vae = vae
        self.log_dir = log_dir
        self.seen = 0
        self.print_every_n_batches = print_every_n_batches
        self.writer = filewriter

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            # summarize filter shapes
            # get filter weights
            # retrieve weights from the second hidden layer
            filters, biases = self.vae.encoder.layers[1].get_weights()
            # normalize filter values to 0-1 so we can visualize them
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            filters = (filters.T.squeeze() * 255.0).astype(np.uint8)
            for map_nr, map in enumerate(filters):
                pil_im = Image.fromarray(map)
                b = BytesIO()
                pil_im.save(b, 'jpeg')
                im_bytes = b.getvalue()

                # create protobuf summary
                im_summary = tf.Summary.Image(encoded_image_string=im_bytes)
                im_summary_value = tf.Summary.Value(tag="layer1 maps/map {}".format(map_nr), image=im_summary)
                summary = tf.Summary(value=[im_summary_value])

                self.writer.add_summary(summary, global_step=self.seen)
                self.writer.flush()


class ReconstructionImagesCallback(Callback):
    """
    Randomly draw 9 Gaussians and reconstruct their images with fixed seeds (consistency over batches)
    """

    def __init__(self, log_dir: str, print_every_n_batches: int, initial_epoch: int, vae: 'VariationalAutoencoder', filewriter):
        """
        
        :param log_dir: the tensorboard log directory
        :param print_every_n_batches: after how many batches this callback will be executed
        :param initial_epoch: the initial epoch
        :param vae: the VariationalAutoencoderModel
        """
        super().__init__()
        self.epoch = initial_epoch
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
        self.seeds = list(range(9))
        self.seen = 0
        self.writer = filewriter

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            for seed in self.seeds:
                # make sure we always reconstruct the same image
                np.random.seed(seed)
                z_new = np.random.normal(size=(1, self.vae.z_dim))
                # predictions are in [0-1] float format
                reconst = self.vae.decoder.predict(np.array(z_new))
                # make byte array
                k = (reconst.squeeze() * 255.0).astype(np.uint8)
                # make pillow image and convert to byte string
                pil_im = Image.fromarray(k)
                b = BytesIO()
                pil_im.save(b, 'jpeg')
                im_bytes = b.getvalue()

                # create protobuf summary
                im_summary = tf.Summary.Image(encoded_image_string=im_bytes)
                im_summary_value = tf.Summary.Value(tag="reconstructions/sample {}".format(seed), image=im_summary)
                summary = tf.Summary(value=[im_summary_value])

                self.writer.add_summary(summary, global_step=self.seen)
                self.writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch += 1


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return new_lr

    return LearningRateScheduler(schedule)
