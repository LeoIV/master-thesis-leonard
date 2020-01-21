from io import BytesIO
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Model
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard
from tensorflow.python.summary.writer.writer import FileWriter


class FeatureMapVisualizationCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VariationalAutoencoder', print_every_n_batches: int,
                 layer_idxs: Sequence[int], x_train: np.ndarray, writer: FileWriter = None,
                 tb_callback: TensorBoard = None):
        super().__init__()
        if tb_callback is None and writer is None:
            raise AttributeError("Either writer or tb_callback has to be set.")
        self.log_dir = log_dir
        self.vae = vae
        self.print_every_n_batches = print_every_n_batches
        self.layer_idxs = layer_idxs
        self.writer = writer
        self.seen = 0
        self.x_train = x_train
        self.tb_callback = tb_callback
        idx = np.random.randint(0, len(self.x_train))
        # draw sample from data
        self.sample = self.x_train[idx]
        # try to convert to 0-255 uint8 array

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
            sample_as_uint8 = self.sample
            if not self.sample.dtype == np.uint8:
                sample_as_uint8 = (self.sample - self.sample.min()) / (self.sample - self.sample.max())
                sample_as_uint8 *= 255.0
                sample_as_uint8 = sample_as_uint8.astype(np.uint8)
            sample_as_uint8 = sample_as_uint8.squeeze()
            image_summary(image=sample_as_uint8, tag="feature_maps_original", global_step=self.seen, writer=self.writer)
            for layer_idx in self.layer_idxs:
                output_layer = self.vae.encoder.layers[layer_idx]
                model = Model(inputs=self.vae.encoder.inputs, outputs=output_layer.output)
                feature_maps = model.predict(self.sample[np.newaxis, :])
                feature_maps = feature_maps.squeeze()
                feature_maps = feature_maps.T
                feature_maps = (feature_maps - feature_maps.min()) / (feature_maps - feature_maps.max())
                feature_maps *= 255.0
                feature_maps = feature_maps.astype(np.uint8)
                for i, map in enumerate(feature_maps):
                    image_summary(image=map, tag="feature_maps_{}/map_{}".format(output_layer.name, i),
                                  global_step=self.seen, writer=self.writer)


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
        self.writer = writer
        self.seen = 0
        self.x_train = x_train
        self.tb_callback = tb_callback
        idx = np.random.randint(0, len(self.x_train))
        # draw sample from data
        self.sample = self.x_train[idx]
        # try to convert to 0-255 uint8 array

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
            sample_as_uint8 = self.sample
            if not self.sample.dtype == np.uint8:
                sample_as_uint8 = (self.sample - self.sample.min()) / (self.sample - self.sample.max())
                sample_as_uint8 *= 255.0
                sample_as_uint8 = sample_as_uint8.astype(np.uint8)
            sample_as_uint8 = sample_as_uint8.squeeze()
            image_summary(image=sample_as_uint8, tag="feature_maps_original", global_step=self.seen, writer=self.writer)
            correlations = []
            for encoder_layer_name, decoder_layer_name in self.layer_mappings:
                encoder_layer = self.encoder_layers[encoder_layer_name]
                truncated_encoder = Model(self.vae.encoder.inputs, encoder_layer.output)
                if encoder_layer_name == self.vae.encoder.layers[0].name:
                    flattened_encoder_output = self.sample[np.newaxis, :].flatten() + 0.001
                else:
                    flattened_encoder_output = truncated_encoder.predict(self.sample[np.newaxis, :]).flatten() + 0.001

                mu = self.no_resample_encoder.predict(self.sample[np.newaxis, :])
                decoder_layer = self.decoder_layers[decoder_layer_name]
                truncated_decoder = Model(self.vae.decoder.inputs, decoder_layer.output)
                flattened_decoder_output = truncated_decoder.predict(mu).flatten() + 0.001

                correlations.append(_correlation_coefficient(flattened_decoder_output, flattened_encoder_output))
            print(correlations)

            summary = tf.Summary(value=[tf.Summary.Value(tag="correlation", simple_value=np.mean(correlations))])

            self.writer.add_summary(summary, global_step=self.seen)
            self.writer.flush()


class KernelVisualizationCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VariationalAutoencoder', print_every_n_batches: int, layer_idx: int,
                 writer: FileWriter = None, tb_callback: TensorBoard = None):
        super().__init__()
        if tb_callback is None and writer is None:
            raise AttributeError("Either writer or tb_callback has to be set.")
        self.vae = vae
        self.log_dir = log_dir
        self.seen = 0
        self.print_every_n_batches = print_every_n_batches
        self.writer = writer
        self.tb_callback = tb_callback
        self.layer_idx = layer_idx

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
            # summarize filter shapes
            # get filter weights
            # retrieve weights from the second hidden layer
            filters, biases = self.vae.encoder.layers[self.layer_idx].get_weights()
            # normalize filter values to 0-1 so we can visualize them
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            filters = (filters.T.squeeze() * 255.0).astype(np.uint8)
            for map_nr, map in enumerate(filters):
                image_summary(image=map, tag="layer1 maps/map {}".format(map_nr), global_step=self.seen,
                              writer=self.writer)


class ReconstructionImagesCallback(Callback):
    """
    Randomly draw 9 Gaussians and reconstruct their images with fixed seeds (consistency over batches)
    """

    def __init__(self, log_dir: str, print_every_n_batches: int, initial_epoch: int, vae: 'VariationalAutoencoder',
                 tb_callback: TensorBoard = None, writer: FileWriter = None):
        """

        :param log_dir:
        :param print_every_n_batches:
        :param initial_epoch:
        :param vae:
        :param tb_callback:
        :param writer:
        """
        super().__init__()
        if tb_callback is None and writer is None:
            raise AttributeError("Either writer or tb_callback has to be set.")
        self.epoch = initial_epoch
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
        self.seeds = list(range(9))
        self.seen = 0
        self.writer = writer
        self.tb_callback = tb_callback

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
            for seed in self.seeds:
                # make sure we always reconstruct the same image
                np.random.seed(seed)
                z_new = np.random.normal(size=(1, self.vae.z_dim))
                # predictions are in [0-1] float format
                reconst = self.vae.decoder.predict(np.array(z_new))
                # make byte array
                k = (reconst.squeeze() * 255.0).astype(np.uint8)

                image_summary(image=k, tag="reconstructions/sample {}".format(seed), global_step=self.seen,
                              writer=self.writer)

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
