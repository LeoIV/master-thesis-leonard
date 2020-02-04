import logging
import os
from io import BytesIO
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Model
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard
from tensorflow.python.summary.writer.writer import FileWriter


class FeatureMapActivationCorrelationCallback(Callback):
    """
    Compute the correlation between
    """

    def __init__(self, log_dir: str, vae: 'VariationalAutoencoder', print_every_n_batches: int,
                 layer_mappings: Sequence[Tuple[str, str]], x_train: np.ndarray, writer: FileWriter = None,
                 tb_callback: TensorBoard = None):
        super().__init__()
        if tb_callback is None and writer is None:
            raise AttributeError("Either writer or tb_callback has to be set.")
        self.log_dir = log_dir
        self.vae = vae
        self.encoder_layers = dict(map(lambda x: (x.name, x), self.vae.encoder.layers))
        self.decoder_layers = dict(map(lambda x: (x.name, x), self.vae.decoder.layers))
        self.print_every_n_batches = print_every_n_batches
        self.layer_mappings = layer_mappings
        self.no_resample_encoder = Model(self.vae.encoder.inputs,
                                         (self.encoder_layers['mu'].output))
        self.seen = 0
        self.x_train = x_train
        idx = np.random.randint(0, len(self.x_train))
        # draw sample from data
        self.sample = self.x_train[idx]
        # try to convert to 0-255 uint8 array
        self.tb_callback = tb_callback
        self.writer = writer

    def set_model(self, model):
        """
        If we have set a tb_callback, we use it's writer for our own summaries.
        :param model:
        :return:
        """
        if self.tb_callback is not None:
            if not self.tb_callback.writer:
                raise AttributeError(
                    "If you  set tb_callback, it has to be set as a Keras callback before this callback!")
            self.writer = self.tb_callback.writer

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            logging.info("Computing activation correlations")
            correlations = []
            for encoder_layer_name, decoder_layer_name in self.layer_mappings:
                encoder_layer = self.encoder_layers[encoder_layer_name]
                truncated_encoder = Model(self.vae.encoder.inputs, encoder_layer.output)
                if encoder_layer_name == self.vae.encoder.layers[0].name:
                    encoder_output = self.sample[np.newaxis, :].squeeze() + 0.001
                else:
                    encoder_output = truncated_encoder.predict(self.sample[np.newaxis, :]).squeeze() + 0.001

                mu = self.no_resample_encoder.predict(self.sample[np.newaxis, :])
                decoder_layer = self.decoder_layers[decoder_layer_name]
                truncated_decoder = Model(self.vae.decoder.inputs, decoder_layer.output)
                decoder_output = truncated_decoder.predict(mu).squeeze() + 0.001

                encoder_output = np.moveaxis(encoder_output, -1, 0)
                decoder_output = np.moveaxis(decoder_output, -1, 0)

                def _key(x):
                    return np.mean(np.sum(x))

                sorted_encoder_maps = sorted(encoder_output, key=_key)
                sorted_decoder_maps = sorted(decoder_output, key=_key)

                flattened_encoder_maps = (np.stack(sorted_encoder_maps)).flatten()
                flattened_decoder_maps = (np.stack(sorted_decoder_maps)).flatten()

                corr = _correlation_coefficient(flattened_encoder_maps, flattened_decoder_maps)
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag="correlation/{}".format(encoder_layer_name), simple_value=corr)])
                self.writer.add_summary(summary, global_step=self.seen)

            print(correlations)
            self.writer.flush()


class ReconstructionImagesCallback(Callback):
    """
    Randomly draw 9 Gaussians and reconstruct their images with fixed seeds (consistency over batches)
    """

    def __init__(self, log_dir: str, print_every_n_batches: int, initial_epoch: int, vae: 'VariationalAutoencoder',
                 num_reconstructions: int = 10):
        """

        :param log_dir:
        :param print_every_n_batches:
        :param initial_epoch:
        :param vae:
        """
        super().__init__()
        self.epoch = initial_epoch
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
        self.seeds = list(range(num_reconstructions))
        self.seen = 0
        self.log_dir = log_dir

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            logging.info("Visualizing reconstructions")
            img_path = os.path.join(self.log_dir, "step_{}".format(self.seen), "reconstructions")
            os.makedirs(img_path, exist_ok=True)
            for seed in self.seeds:
                # make sure we always reconstruct the same image
                np.random.seed(seed)
                z_new = np.random.normal(size=(1, self.vae.z_dim))
                np.random.seed(None)
                # predictions are in [0-1] float format
                reconst = self.vae.decoder.predict(np.array(z_new))
                # make byte array
                k = (reconst.squeeze() * 255.0).astype(np.uint8)

                Image.fromarray(k.squeeze()).save(os.path.join(img_path, "sample_{}.jpg".format(seed)))

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


def image_summary(image: np.ndarray, tag: str, global_step: int, writer: FileWriter):
    """
    Wrapper function to create a tensorboard image summary from an np.ndarray.
    :param image: the image data. dtype has to be np.uint8 and shape has to be either [h,w,c] or [h,w] (grayscale) with h=height, w=width, c=channels
    :param tag: the tag to use for the summary. If you want images to be grouped in a tab in tensorboard use something like "group_name/tag_name"
    :param global_step: the global step (nr of the summary)
    :param writer: the tensorboard filewriter
    :return: nothing
    """
    assert image.dtype == np.uint8
    # make pillow image and convert to byte string
    pil_im = Image.fromarray(image)
    b = BytesIO()
    pil_im.save(b, 'jpeg')
    im_bytes = b.getvalue()

    # create protobuf summary
    im_summary = tf.Summary.Image(encoded_image_string=im_bytes)
    im_summary_value = tf.Summary.Value(tag=tag, image=im_summary)
    summary = tf.Summary(value=[im_summary_value])

    writer.add_summary(summary, global_step=global_step)
    writer.flush()


def _correlation_coefficient(a, b):
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    a_centered = a - mean_a
    b_centered = b - mean_b
    return np.dot(a_centered, b_centered) / np.sqrt(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered))
