import math
import os
import pickle
from typing import Sequence, Union, Tuple

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras_preprocessing.image import DirectoryIterator, Iterator
from receptivefield.keras import KerasReceptiveField

from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from models.ModelWrapper import ModelWrapper
from utils.callbacks import ReconstructionImagesCallback, step_decay_schedule, \
    FeatureMapActivationCorrelationCallback, ActivationVisualizationCallback


class VariationalAutoencoder(ModelWrapper):
    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size: Sequence[Union[int, Tuple[int, int]]],
                 encoder_conv_strides: Sequence[Union[int, Tuple[int, int]]], decoder_conv_t_filters,
                 decoder_conv_t_kernel_size: Sequence[Union[int, Tuple[int, int]]],
                 decoder_conv_t_strides: Sequence[Union[int, Tuple[int, int]]], z_dim: int, log_dir: str,
                 use_batch_norm: bool = False,
                 use_dropout: bool = False):

        self.name = 'variational_autoencoder'

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self.log_dir = log_dir

        layer_names = ["encoder_conv_{}".format(i) for i in range(self.n_layers_encoder)]
        self.rfs = KerasReceptiveField(self._build).compute(input_shape=self.input_dim[0:2],
                                                            input_layer="encoder_input",
                                                            output_layers=layer_names)
        self.rfs = dict(map(lambda x: (x[0], x[1]), zip(layer_names, self.rfs)))

    def _build(self, input_dim):

        # THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):

            if i == 0:
                conv_layer = Conv2D(input_shape=self.input_dim,
                                    filters=self.encoder_conv_filters[i], kernel_size=self.encoder_conv_kernel_size[i],
                                    strides=self.encoder_conv_strides[i], padding='same', name='encoder_conv_' + str(i)
                                    )
            else:
                conv_layer = Conv2D(
                    filters=self.encoder_conv_filters[i], kernel_size=self.encoder_conv_kernel_size[i],
                    strides=self.encoder_conv_strides[i], padding='same', name='encoder_conv_' + str(i)
                )

            x = conv_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)

        # THE DECODER

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i], kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i], padding='same', name='decoder_conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        # THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)
        return self.encoder

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        # COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(os.path.join(folder, 'visualizations')):
            os.makedirs(os.path.join(folder, 'visualizations'))
        if not os.path.exists(os.path.join(folder, 'weights')):
            os.makedirs(os.path.join(folder, 'weights'))
        if not os.path.exists(os.path.join(folder, 'images')):
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim, self.encoder_conv_filters, self.encoder_conv_kernel_size, self.encoder_conv_strides,
                self.decoder_conv_t_filters, self.decoder_conv_t_kernel_size, self.decoder_conv_t_strides, self.z_dim,
                self.use_batch_norm, self.use_dropout
            ], f)

        self.plot_model(folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, training_data, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0,
              lr_decay=1, embedding_samples: int = 5000):

        if isinstance(training_data, Iterator):
            training_data: DirectoryIterator
            n_batches = embedding_samples // batch_size
            if n_batches > training_data.n:
                n_batches = training_data.n
            samples = []
            for i in range(n_batches):
                samples.append(training_data.next()[0])
            embeddings_data = np.concatenate(samples, axis=0)
            training_data.reset()

        else:
            embeddings_data = training_data[:5000]

        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)
        tb_callback = TensorBoard(log_dir=self.log_dir, batch_size=batch_size, embeddings_freq=1, update_freq="batch",
                                  embeddings_layer_names=["mu"], embeddings_data=embeddings_data)
        custom_callback = ReconstructionImagesCallback(log_dir='./logs', print_every_n_batches=print_every_n_batches,
                                                       initial_epoch=initial_epoch, vae=self)
        kv_callback = KernelVisualizationCallback(log_dir=self.log_dir, vae=self,
                                                  print_every_n_batches=print_every_n_batches,
                                                  layer_idx=1)
        fm_callback = FeatureMapVisualizationCallback(log_dir=self.log_dir, model_wrapper=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idxs=[2, 22, 4, 20, 6, 18, 8, 16],
                                                      x_train=embeddings_data)
        av_callback = ActivationVisualizationCallback(log_dir=self.log_dir, model_wrapper=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idxs=[2, 4, 6, 8])
        fma_callback = FeatureMapActivationCorrelationCallback(log_dir=self.log_dir, vae=self,
                                                               print_every_n_batches=print_every_n_batches,
                                                               layer_mappings=[("encoder_input", "activation_1"),
                                                                               ("encoder_conv_0", "decoder_conv_t_2"),
                                                                               ("leaky_re_lu_1", "leaky_re_lu_7"),
                                                                               ("encoder_conv_1", "decoder_conv_t_1"),
                                                                               ("leaky_re_lu_2", "leaky_re_lu_6"),
                                                                               ("encoder_conv_2", "decoder_conv_t_0"),
                                                                               ("leaky_re_lu_3", "leaky_re_lu_5"),
                                                                               ("encoder_conv_3", "reshape_1")],
                                                               x_train=embeddings_data,
                                                               tb_callback=tb_callback)
        # tb_callback has to be first as we use its filewriter subsequently but it is initialized by keras in this given order
        callbacks_list = [checkpoint1, checkpoint2, tb_callback, fm_callback, kv_callback, fma_callback,
                          custom_callback, lr_sched]

        print("Training for {} epochs".format(epochs))

        if isinstance(training_data, Iterator):
            steps_per_epoch = math.ceil(training_data.n / batch_size)
            self.model.fit_generator(
                training_data, shuffle=True, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list,
                steps_per_epoch=steps_per_epoch
            )
        else:
            self.model.fit(
                training_data, training_data, batch_size=batch_size, shuffle=False, epochs=epochs,
                callbacks=callbacks_list
            )
        print("Training finished")

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'visualizations/model.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(run_folder, 'visualizations/encoder.png'), show_shapes=True,
                   show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(run_folder, 'visualizations/decoder.png'), show_shapes=True,
                   show_layer_names=True)
