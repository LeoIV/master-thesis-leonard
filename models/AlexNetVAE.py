import math
from typing import Tuple

import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization, Lambda, \
    Reshape, Conv2DTranspose, Activation, LeakyReLU, UpSampling2D, ReLU
from keras.models import Model

from models.model_abstract import VAEWrapper


class AlexNetVAE(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, z_dim: int, kernel_visualization_layer: int = -1,
                 use_batch_norm: bool = False, use_dropout: bool = False, dropout_rate: float = 0.5,
                 feature_map_layers=None, num_samples: int = 5, inner_activation: str = "ReLU", use_fc: bool = True,
                 decay_rate: float = 1e-7, feature_map_reduction_factor: int = 1):

        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor)
        self.name = 'variational_autoencoder'

        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_fc = use_fc

        self._build()

    def _build(self):

        # THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='model_input')

        x = encoder_input

        # Layer 1
        x = Conv2D(filters=math.ceil(96 / self.feature_map_reduction_factor), input_shape=(224, 224, 3),
                   kernel_size=(11, 11), strides=(4, 4), padding="same")(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(filters=math.ceil(256 / self.feature_map_reduction_factor), kernel_size=(5, 5), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(filters=math.ceil(384 / self.feature_map_reduction_factor), kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 4
        x = Conv2D(filters=math.ceil(384 / self.feature_map_reduction_factor), kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 5
        x = Conv2D(filters=math.ceil(256 / self.feature_map_reduction_factor), kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 6
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        shape_before_flattening = K.int_shape(x)[1:]
        # Flatten
        x = Flatten()(x)

        if self.use_fc:
            # FC1
            x = Dense(math.ceil(4096 / self.feature_map_reduction_factor))(x)
            # , input_shape=(np.prod(self.input_dim),)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=self.dropout_rate)(x)

            # FC2
            x = Dense(math.ceil(4096 / self.feature_map_reduction_factor))(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=self.dropout_rate)(x)

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

        x = decoder_input

        if self.use_fc:
            # FC2 - reverse
            x = Dense(math.ceil(4096 / self.feature_map_reduction_factor))(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=self.dropout_rate)(x)
            # FC1 - reverse
            x = Dense(np.prod(shape_before_flattening))(x)
            x = LeakyReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=self.dropout_rate)(x)

        # FC1 - reverse
        x = Dense(np.prod(shape_before_flattening))(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)

        # Unflatten
        x = Reshape(shape_before_flattening)(x)

        # Layer 6 - reverse
        # x = UpSampling2D(size=(2, 2))(x)

        # Layer 5 - reverse
        x = Conv2DTranspose(filters=math.ceil(384 / self.feature_map_reduction_factor), kernel_size=(3, 3),
                            strides=(2, 2), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 4 - reverse
        x = Conv2DTranspose(filters=math.ceil(384 / self.feature_map_reduction_factor), kernel_size=(3, 3),
                            padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 3 - reverse
        x = Conv2DTranspose(filters=math.ceil(256 / self.feature_map_reduction_factor), kernel_size=(3, 3),
                            padding='same', strides=(2, 2))(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 2 - reverse
        x = Conv2DTranspose(filters=math.ceil(96 / self.feature_map_reduction_factor), kernel_size=(5, 5),
                            padding='same', strides=(2, 2))(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 1 - revese
        # x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(filters=self.input_dim[-1], kernel_size=(11, 11), strides=(4, 4), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        decoder_output = x = Activation('sigmoid')(x)

        self.decoder = Model(decoder_input, decoder_output)

        # THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)
        return self.encoder
