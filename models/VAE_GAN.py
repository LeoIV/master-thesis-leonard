import logging
import math
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Tuple, Sequence, Union, Optional

import numpy as np
from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Reshape, \
    Activation, Lambda, Conv2DTranspose, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras_preprocessing.image import Iterator, DirectoryIterator
from tqdm import tqdm

from callbacks.ActivationVisualizationCallback import ActivationVisualizationCallback
from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.HiddenSpaceCallback import HiddenSpaceCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from callbacks.LossLoggingCallback import LossLoggingCallback
from callbacks.ReconstructionCallback import ReconstructionImagesCallback
from models.model_abstract import VAEWrapper
from utils.vae_utils import sampling


class VAEGAN(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int],
                 encoder_conv_filters,
                 encoder_conv_kernel_size: Sequence[Union[int, Tuple[int, int]]],
                 encoder_conv_strides: Sequence[Union[int, Tuple[int, int]]],
                 decoder_conv_t_filters,
                 decoder_conv_t_kernel_size: Sequence[Union[int, Tuple[int, int]]],
                 decoder_conv_t_strides: Sequence[Union[int, Tuple[int, int]]],
                 z_dims: Sequence[int],
                 log_dir: str,
                 feature_map_layers: Sequence[int], kernel_visualization_layer: int, dropout_rate: float,
                 use_batch_norm: bool = False, use_dropout: bool = False, num_samples: int = 10,
                 inner_activation: str = "ReLU", decay_rate: float = 1e-7, feature_map_reduction_factor: int = 1):
        """
        Variational autoencoder with adversarial loss, implementation based on
        https://arxiv.org/pdf/1512.09300.pdf and
        https://github.com/baudm/vaegan-celebs-keras/tree/09a012bfecd8d0b202b5531bc646100428d3fa83

        :param input_dim: dimensionality of the input image
        :param use_dropout:
        :param use_batch_norm:
        :param log_dir:
        :param kernel_visualization_layer: layer index of the layer to visualize the kernels
        :param num_samples: number of generations/reconstructions during training
        :param feature_map_layers: layer indices of layers to visualize feature maps during training
        :param inner_activation: either "ReLU" or "LeakyReLU"
        :param decay_rate: learning rate decay rate
        :param feature_map_reduction_factor: factor by which to reduce feature maps or number of units
        :param z_dims: dimensionality of latent spaces (list of three ints in this case)
        :param dropout_rate:
        """
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dims, ["mu"], ["log_var"])
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.encoder_conv_strides = encoder_conv_strides
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_filters = encoder_conv_filters
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self.dropout_rate = dropout_rate

        def _discriminator(input_shape: Tuple[int, int, int]):
            x = inpt = Input(shape=input_shape, name="discriminator_input")
            x = Conv2D(batch_input_shape=input_shape, filters=64, kernel_size=4, strides=2, padding='same')(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = x_feat = Conv2D(batch_input_shape=input_shape, filters=128, kernel_size=4, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            if self.input_dim[0] >= 60:
                x = x_feat = Conv2D(batch_input_shape=input_shape, filters=256, kernel_size=4, strides=2,
                                    padding='same')(x)
                x = BatchNormalization()(x)
            if self.input_dim[0] >= 100:
                x = LeakyReLU(alpha=0.2)(x)
                x = Conv2D(batch_input_shape=input_shape, filters=512, kernel_size=4, strides=2, padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(batch_input_shape=input_shape, filters=1, kernel_size=1, strides=4, padding='same')(x)
            x = Flatten()(x)
            x = Dense(1)(x)
            x = Activation(activation='sigmoid')(x)
            return Model(inpt, [x_feat, x], name="vlae_gan_discriminator")

        self.discriminator = _discriminator(self.input_dim)

        self._build()

    def _build(self):

        # THE ENCODER
        self.inputs = Input(shape=self.input_dim, name='encoder_input')

        x = self.inputs

        for i in range(self.n_layers_encoder):

            if i == 0:
                conv_layer = Conv2D(input_shape=self.input_dim,
                                    filters=math.ceil(self.encoder_conv_filters[i] / self.feature_map_reduction_factor),
                                    kernel_size=self.encoder_conv_kernel_size[i],
                                    strides=self.encoder_conv_strides[i], padding='same', name='encoder_conv_' + str(i)
                                    )
            else:
                conv_layer = Conv2D(
                    filters=math.ceil(self.encoder_conv_filters[i] / self.feature_map_reduction_factor),
                    kernel_size=self.encoder_conv_kernel_size[i],
                    strides=self.encoder_conv_strides[i], padding='same', name='encoder_conv_' + str(i)
                )

            x = conv_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.z_dims[0], name='mu')(x)
        self.log_var = Dense(self.z_dims[0], name='log_var')(x)

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(self.inputs, encoder_output, name='encoder')

        # THE DECODER

        decoder_input = Input(shape=(self.z_dims[0],), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=math.ceil(self.decoder_conv_t_filters[i] / self.feature_map_reduction_factor),
                kernel_size=self.decoder_conv_t_kernel_size[i], use_bias=False,
                strides=self.decoder_conv_t_strides[i], padding='same', name='decoder_conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        self.decoder = Model(decoder_input, x)
        decoder_output = self.decoder(encoder_output)

        self.model = Model(self.inputs, decoder_output, name='vae_gan')

        #####

        self.x_tilde = self.decoder(encoder_output)
        self.x_p = self.decoder(self.decoder.inputs)

        self.dis_x_feat, self.dis_x = self.discriminator(self.inputs)
        self.dis_x_feat_tilde, self.dis_x_tilde = self.discriminator(self.x_tilde)
        self.dis_x_feat_p, self.dis_x_p = self.discriminator(self.x_p)

        return self.encoder

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        def mean_gaussian_negative_log_likelihood(y_true, y_pred):
            nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
            axis = tuple(range(1, len(K.int_shape(y_true))))
            return K.mean(K.sum(nll, axis=axis), axis=-1)

        def vae_kl_loss():
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
            return kl_loss

        dis_nl_loss = mean_gaussian_negative_log_likelihood(self.dis_x_feat, self.dis_x_feat_tilde)
        kl_loss = vae_kl_loss()

        self.encoder_train = Model(self.inputs, self.dis_x_feat_tilde)
        self.encoder_train.add_loss(10 * kl_loss)
        self.encoder_train.add_loss(dis_nl_loss)

        self.decoder_train = Model([self.inputs, *self.decoder.inputs], [self.dis_x_tilde, self.dis_x_p])
        self.decoder_train.add_loss(0.75 * dis_nl_loss)

        self.discriminator_train = Model([self.inputs, *self.decoder.inputs],
                                         [self.dis_x, self.dis_x_tilde, self.dis_x_p])

        self.vlae_gan = Model(self.inputs, self.dis_x_tilde)

    def train(self, x_train: Union[Iterator, np.ndarray], batch_size, epochs, weights_folder, print_every_n_batches=100,
              initial_epoch=0, lr_decay=1, embedding_samples: int = 5000, y_train: Optional[np.ndarray] = None,
              x_test: Optional[Union[Iterator, np.ndarray]] = None, y_test: Optional[np.ndarray] = None,
              steps_per_epoch: int = None, **kwargs):
        """

        :param x_train:
        :param batch_size: best batch size for this model: 128
        :param epochs:
        :param weights_folder:
        :param print_every_n_batches:
        :param initial_epoch:
        :param lr_decay:
        :param embedding_samples:
        :param y_train:
        :param x_test:
        :param y_test:
        :param steps_per_epoch:
        :param kwargs:
        :return:
        """

        embedding_callback_params = kwargs.get('embedding_callback_params', {})

        if isinstance(x_train, Iterator):
            x_train: DirectoryIterator
            n_batches = embedding_samples // batch_size
            if n_batches > x_train.n:
                n_batches = x_train.n
            samples = []
            for i in range(n_batches):
                samples.append(x_train.next()[0])
            x_train_subset = np.concatenate(samples, axis=0)
            x_train.reset()
            y_train = None
            y_embedding = None

        else:
            x_train_subset = x_train[:5000]
            y_embedding = y_train[:5000] if y_train is not None else None

        # checkpoint2 = ModelCheckpoint(os.path.join(weights_folder, 'weights.h5'), save_weights_only=True, verbose=1)

        executor = ThreadPoolExecutor(max_workers=5)
        kv_encoder_callback = KernelVisualizationCallback(log_dir=self.log_dir,
                                                          print_every_n_batches=print_every_n_batches,
                                                          model=self.encoder, model_name="encoder", executor=executor)
        kv_decoder_callback = KernelVisualizationCallback(log_dir=self.log_dir,
                                                          print_every_n_batches=print_every_n_batches,
                                                          model=self.decoder, model_name="decoder", executor=executor)
        fm_encoder_callback = FeatureMapVisualizationCallback(log_dir=self.log_dir, model=self.encoder,
                                                              model_name="encoder",
                                                              print_every_n_batches=print_every_n_batches,
                                                              x_train=x_train_subset, executor=executor)
        fm_decoder_callback = FeatureMapVisualizationCallback(log_dir=self.log_dir, model=self.decoder,
                                                              model_name="decoder",
                                                              print_every_n_batches=print_every_n_batches,
                                                              x_train=x_train_subset,
                                                              x_train_transform=lambda x, m: m.predict(x),
                                                              transform_params=[self.encoder], executor=executor)
        rc_callback = ReconstructionImagesCallback(log_dir=self.log_dir, print_every_n_batches=print_every_n_batches,
                                                   initial_epoch=initial_epoch, vae=self, x_train=x_train_subset,
                                                   num_reconstructions=self.num_samples, num_inputs=len(self.z_dims))
        av_encoder_callback = ActivationVisualizationCallback(log_dir=self.log_dir, model=self.encoder,
                                                              model_name='encoder',
                                                              x_train=x_train_subset,
                                                              print_every_n_batches=print_every_n_batches)
        av_decoder_callback = ActivationVisualizationCallback(log_dir=self.log_dir, model=self.decoder,
                                                              model_name='decoder',
                                                              x_train=x_train_subset,
                                                              print_every_n_batches=print_every_n_batches,
                                                              x_train_transform=lambda x, m: m.predict(x),
                                                              transform_params=[self.encoder])
        hs_callback = HiddenSpaceCallback(log_dir=self.log_dir, vae=self, batch_size=batch_size,
                                          x_train=x_train_subset, y_train=y_embedding, max_samples=5000,
                                          layer_names=self.mu_layer_names, plot_params=embedding_callback_params,
                                          executor=executor)
        ll_callback = LossLoggingCallback(logdir=self.log_dir)
        # tb_callback has to be first as we use its filewriter subsequently but it is initialized by keras in this given order
        callbacks_list = [kv_encoder_callback, kv_decoder_callback, fm_encoder_callback, fm_decoder_callback,
                          hs_callback, av_encoder_callback, av_decoder_callback, rc_callback, ll_callback]

        logging.info("Training for {} epochs".format(epochs))

        optimizer = Adam(lr=self.learning_rate, beta_1=0.5, decay=self.decay_rate)

        def set_trainable(model: Model, trainable: bool):
            """
            Set layers of model to trainable / not trainable as specified by 'trainable' (in place)
            :param model: the model to manipulate
            :param trainable:
            :return: nothing
            """
            model.trainable = trainable
            for layer in model.layers:
                layer.trainable = trainable

        set_trainable(self.encoder, False)
        set_trainable(self.decoder, False)
        set_trainable(self.discriminator, True)
        self.discriminator_train.compile(optimizer, ['binary_crossentropy'] * 3, ['acc'] * 3)
        self.discriminator_train.summary()

        set_trainable(self.discriminator, False)
        set_trainable(self.decoder, True)
        self.decoder_train.compile(optimizer, ['binary_crossentropy'] * 2, ['acc'] * 2)
        self.decoder_train.summary()

        set_trainable(self.decoder, False)
        set_trainable(self.encoder, True)
        self.encoder_train.compile(optimizer)
        self.encoder_train.summary()

        set_trainable(self.vlae_gan, True)

        models = [self.discriminator_train, self.decoder_train, self.encoder_train]
        epoch = initial_epoch
        for cb in callbacks_list:
            cb.set_model(self.model)
            cb.on_train_begin()
        if x_test is not None:
            x_test = np.copy(x_test)
            logging.info("Evaluating models")
            if isinstance(x_test, Iterator):
                raise RuntimeError("Model evaluation currently is only supported for numpy test data.")
            assert isinstance(x_test, np.ndarray)
            # make sure len(x_test) is multiple of batch_size
            test_batch_size = 32
            test_size = x_test.shape[0]
            num_batches = test_size // test_batch_size
            x_test = x_test[:num_batches * test_batch_size]
            test_size = x_test.shape[0]
        while epoch < epochs:
            batch_index = 0
            for cb in callbacks_list:
                cb.on_epoch_begin(epoch)

            if isinstance(x_train, Iterator):
                steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else x_train.n // batch_size
            else:
                steps_per_epoch = len(x_train) // batch_size
            with tqdm(total=steps_per_epoch,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | {postfix[0]} {postfix[1][value]} | {postfix[2]} {postfix[3][value]:.3f} | {postfix[4]} {postfix[5][value]:.3f}",
                      postfix=["Discriminator", dict(value=0), "Decoder", dict(value=0), "Encoder",
                               dict(value=0)]) as t:
                for steps_done in range(steps_per_epoch):
                    for cb in callbacks_list:
                        cb.on_batch_begin(batch_index, {})
                    losses = {}
                    for model in models:
                        if isinstance(x_train, Iterator):
                            x, y = next(x_train)
                        else:
                            x, y = (x_train[batch_index * batch_size:batch_index * batch_size + batch_size],
                                    y_train[batch_index * batch_size:batch_index * batch_size + batch_size])
                        if model == self.discriminator_train:
                            x = [x, *[np.random.normal(size=(batch_size, z)) for z in self.z_dims]]
                            y_real = np.full((batch_size,), fill_value=0.9, dtype='float32')
                            y_fake = np.zeros((batch_size,), dtype='float32')
                            y = [y_real, y_fake, y_fake]
                            outs = model.train_on_batch(x, y)
                            t.postfix[1]["value"] = "{:.3f} {:.3f} {:.3f}".format(outs[4],
                                                                                  outs[7 if len(outs) > 7 else 5],
                                                                                  outs[10 if len(outs) > 7 else 6])
                            losses['discriminator_loss'] = outs[0]
                            losses['discriminator_loss_x_true'] = outs[1]
                            losses['discriminator_loss_x_reconstr'] = outs[2]
                            losses['discriminator_loss_x_sampling'] = outs[3]
                            losses['discriminator_acc_x_true'] = outs[4]
                            losses['discriminator_acc_x_reconstr'] = outs[7 if len(outs) > 7 else 5]
                            losses['discriminator_acc_x_sampling'] = outs[10 if len(outs) > 7 else 6]
                        elif model == self.decoder_train:
                            x = [x, *[np.random.normal(size=(batch_size, z)) for z in self.z_dims]]
                            y_real = np.ones((batch_size,), dtype='float32')
                            y = [y_real, y_real]
                            outs = model.train_on_batch(x, y)
                            t.postfix[3]["value"] = outs[3]
                            losses['decoder_acc'] = outs[3]
                            losses['decoder_loss'] = outs[0]
                        elif model == self.encoder_train:
                            outs = model.train_on_batch(x, None)
                            mean_loss = np.mean(outs)
                            t.postfix[5]["value"] = mean_loss
                            losses['encoder_loss'] = mean_loss
                    t.update()
                    for cb in callbacks_list:
                        cb.on_batch_end(batch_index, losses)
                    batch_index += 1
            losses = {}
            if x_test is not None:
                y_real = np.ones((test_size,), dtype='float32')
                y_fake = np.zeros((test_size,), dtype='float32')
                for model in models:
                    if model == self.discriminator_train:
                        x = [x_test, *[np.random.normal(size=(test_size, z)) for z in self.z_dims]]
                        y = [y_real, y_fake, y_fake]
                        outs = model.evaluate(x, y, batch_size=test_batch_size)
                        losses['discriminator_loss'] = outs[0]
                        losses['discriminator_loss_x_true'] = outs[1]
                        losses['discriminator_loss_x_reconstr'] = outs[2]
                        losses['discriminator_loss_x_sampling'] = outs[3]
                        losses['discriminator_acc_x_true'] = outs[4]
                        losses['discriminator_acc_x_reconstr'] = outs[7 if len(outs) > 7 else 5]
                        losses['discriminator_acc_x_sampling'] = outs[10 if len(outs) > 7 else 6]
                    elif model == self.decoder_train:
                        x = [x_test, *[np.random.normal(size=(test_size, z)) for z in self.z_dims]]
                        y = [y_real, y_real]
                        outs = model.evaluate(x, y, batch_size=test_batch_size)
                        losses['decoder_acc'] = outs[3]
                        losses['decoder_loss'] = outs[0]
                    elif model == self.encoder_train:
                        outs = model.evaluate(x_test, None, batch_size=test_batch_size)
                        mean_loss = np.mean(outs)
                        losses['encoder_loss'] = mean_loss
            for cb in callbacks_list:
                cb.on_epoch_end(epoch, losses)
                self.model.save(os.path.join(self.log_dir, 'weights.h5'))
                self.model.save(os.path.join(self.log_dir, 'weights_epoch_{}.h5'.format(epoch)))
            epoch += 1

        for cb in callbacks_list:
            cb.on_train_end()
