import logging
import os
from io import BytesIO
from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from keras import Model, losses
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard
from keras.layers import Conv2D, LeakyReLU, Input
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from tensorflow.python.summary.writer.writer import FileWriter
from vis.visualization import get_num_filters, visualize_activation


class FeatureMapVisualizationCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VariationalAutoencoder', print_every_n_batches: int,
                 layer_idxs: Sequence[int], x_train: np.ndarray, num_samples: int = 5):
        super().__init__()
        plt.rcParams["figure.figsize"] = (8 * num_samples, 100)
        self.log_dir = log_dir
        self.vae = vae
        self.print_every_n_batches = print_every_n_batches
        self.layer_idxs = layer_idxs
        self.seen = 0
        self.x_train = x_train
        self.fmas = {}
        self.batch_nrs = []
        idxs = np.random.randint(0, len(self.x_train), num_samples)
        self.samples = [np.copy(self.x_train[idx]) for idx in idxs]

    @staticmethod
    def _save_feature_maps(img_path_layer, fms):
        feature_maps = np.copy(fms).squeeze()
        feature_maps = np.moveaxis(feature_maps, [0, 1, 2], [1, 2, 0])
        feature_maps = (feature_maps - feature_maps.min()) / (feature_maps - feature_maps.max())
        feature_maps *= 255.0
        feature_maps = feature_maps.astype(np.uint8)
        for i, f_map in enumerate(feature_maps):
            Image.fromarray(f_map.squeeze()).save(os.path.join(img_path_layer, "map_{}.jpg".format(i)))

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            self.batch_nrs.append(batch)
            fig, ax = plt.subplots(1 + len(self.layer_idxs), len(self.samples))
            for sample_nr, sample in enumerate(self.samples):

                # draw sample from data
                sample_as_uint8 = np.copy(sample)

                if not sample_as_uint8.dtype == np.uint8:
                    sample_as_uint8 *= 255.0
                    sample_as_uint8 = sample_as_uint8.astype(np.uint8)
                sample_as_uint8 = sample_as_uint8.squeeze()
                img_path = os.path.join(self.log_dir, "step_{}".format(self.seen), "feature_map",
                                        "sample_{}".format(sample_nr))
                os.makedirs(img_path, exist_ok=True)
                ax[0, sample_nr].imshow(sample.squeeze(), cmap='gray')
                Image.fromarray(sample_as_uint8).save(
                    os.path.join(img_path, "original.jpg"))
                for i, layer_idx in enumerate(self.layer_idxs):
                    logging.info("Visualizing feature maps for layer {}".format(layer_idx))
                    output_layer = self.vae.encoder.layers[layer_idx]
                    img_path_layer = os.path.join(img_path, "{}".format(output_layer.name))
                    os.makedirs(img_path_layer, exist_ok=True)
                    model = Model(inputs=self.vae.encoder.inputs, outputs=output_layer.output)
                    feature_maps = model.predict(sample[np.newaxis, :])
                    self._save_feature_maps(img_path_layer, feature_maps)
                    fmas = np.sum(np.abs(feature_maps), axis=tuple(range(len(feature_maps.shape)))[:-1])
                    fmas = fmas
                    self.fmas.setdefault(sample_nr, {})
                    self.fmas[sample_nr].setdefault(layer_idx, [])
                    self.fmas[sample_nr][layer_idx].append(fmas)
                    fmas_stacked = np.copy(np.stack(self.fmas[sample_nr][layer_idx]))
                    fmas_stacked /= np.max(fmas_stacked)
                    ax[i + 1, sample_nr].set_title('Normalized activations - Layer {}'.format(output_layer.name))
                    ax[i + 1, sample_nr].imshow(fmas_stacked, origin='lower', interpolation="none")
                    ax[i + 1, sample_nr].set(xlabel="#feature map", ylabel="# batch")
                    ax[i + 1, sample_nr].set_yticks(np.arange(len(self.batch_nrs), step=2), self.batch_nrs[0::2])
            plt.colorbar(ax[1, 0].get_images()[0])
            plt.savefig(os.path.join(self.log_dir, "step_{}".format(self.seen), "feature_map", 'activations.png'))
            plt.close()


class ActivationVisualizationCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VariationalAutoencoder', print_every_n_batches: int,
                 layer_idxs: Sequence[int]):
        super().__init__()

        self.log_dir = log_dir
        self.vae = vae
        self.print_every_n_batches = print_every_n_batches
        self.layer_idxs = layer_idxs
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0 and batch > 0:
            self.seen += 1
            for i, layer_idx in enumerate(self.layer_idxs):
                # get current target layer
                layer = self.vae.encoder.layers[layer_idx]
                if layer.name not in self.vae.rfs:
                    raise AttributeError(
                        "layers of which you want to visualize the optimal stimuli have to have a defined receptive field in self.rfs")
                # use layer receptive field size as input size
                # we assume quadratic receptive fields
                logging.info("Visualizing max activations for layer {}".format(layer.name))
                input_size = max(self.vae.rfs[layer.name].rf.size[0:2])
                input_size = [input_size, input_size, self.vae.encoder.input.shape[-1].value]
                inp = x = Input(shape=input_size)
                for l in self.vae.encoder.layers[1:]:
                    if isinstance(l, Conv2D):
                        x = Conv2D(filters=l.filters, kernel_size=l.kernel_size, strides=l.strides, trainable=False)(x)
                    elif isinstance(l, LeakyReLU):
                        x = LeakyReLU(alpha=l.alpha, trainable=False)(x)
                    else:
                        raise ValueError("only Conv2D or LeakyReLU layers supported.")
                    if l.name == layer.name:
                        break
                truncated_encoder = Model(inp, x)
                for j, _ in enumerate(truncated_encoder.layers):
                    if j == 0:
                        continue
                    # copy weights for new model
                    truncated_encoder.layers[j].set_weights(self.vae.encoder.layers[j].get_weights())

                truncated_encoder.compile(optimizer=Adam(lr=self.vae.learning_rate), loss=losses.mean_squared_error)

                filters = list(range(get_num_filters(truncated_encoder.layers[layer_idx])))
                img_path = os.path.join(self.log_dir, "step_{}".format(self.seen), "max_activations",
                                        "layer_{}".format(layer_idx))
                os.makedirs(img_path, exist_ok=True)
                for f_idx in filters:
                    img = visualize_activation(truncated_encoder, layer_idx, filter_indices=f_idx, lp_norm_weight=0.01,
                                               tv_weight=0.01)
                    '''img = visualize_activation(self.vae.encoder, layer_idx, filter_indices=idx,
                                               seed_input=img,
                                               input_modifiers=[Jitter(0.5)])'''
                    img = Image.fromarray(img.squeeze()).resize((100, 100), resample=Image.NEAREST)
                    draw = ImageDraw.Draw(img)
                    draw.text((1, 1), "L{}F{}".format(layer_idx, f_idx), color=0)
                    img.save(os.path.join(img_path, "filter_{}.jpg".format(f_idx)))


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
                    return np.sum(np.abs(x))

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


class KernelVisualizationCallback(Callback):
    def __init__(self, log_dir: str, vae: 'VariationalAutoencoder', print_every_n_batches: int, layer_idx: int):
        super().__init__()
        self.vae = vae
        self.log_dir = log_dir
        self.seen = 0
        self.print_every_n_batches = print_every_n_batches
        self.layer_idx = layer_idx

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1
            logging.info("Visualizing kernels")
            # summarize filter shapes
            # get filter weights
            # retrieve weights from the second hidden layer
            filters, biases = self.vae.encoder.layers[self.layer_idx].get_weights()
            # normalize filter values to 0-1 so we can visualize them
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            filters = np.moveaxis(filters, (0, 1), (-2, -1))
            filters = (filters.reshape(((-1,) + filters.shape[-2:])) * 255.0).astype(np.uint8)
            img_path = os.path.join(self.log_dir, "step_{}".format(self.seen), "layer1_kernels")
            os.makedirs(img_path, exist_ok=True)
            for map_nr, map in enumerate(filters):
                Image.fromarray(map.squeeze()).save(os.path.join(img_path, "map_{}.jpg".format(map_nr)))


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
        :param tb_callback:
        :param writer:
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
