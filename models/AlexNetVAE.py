import logging
import math
import os
from typing import Tuple

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization, Lambda, \
    Reshape, Conv2DTranspose, Activation, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import DirectoryIterator, Iterator

from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from callbacks.LossLoggingCallback import LossLoggingCallback
from models.model_abstract import VAEWrapper
from utils.callbacks import step_decay_schedule, ReconstructionImagesCallback


class AlexNetVAE(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, z_dim: int, kernel_visualization_layer: int = -1,
                 use_batch_norm: bool = False, use_dropout: bool = False, dropout_rate: float = 0.5,
                 feature_map_layers=None, num_samples: int = 5):

        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers)
        self.name = 'variational_autoencoder'

        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self._build(self.input_dim)

    def _build(self, input_dim):

        # THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='model_input')

        x = encoder_input

        # Layer 1
        x = Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(filters=256, kernel_size=(5, 5), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 4
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 5
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 6
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        shape_before_flattening = K.int_shape(x)[1:]
        # Flatten
        x = Flatten()(x)

        # FC1
        x = Dense(4096)(x)
        # , input_shape=(np.prod(self.input_dim),)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)

        # FC2
        x = Dense(4096)(x)
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

        # FC2 - reverse
        x = Dense(4096)(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)
        # FC1 - reverse
        x = Dense(np.prod(shape_before_flattening))(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)

        # Unflatten
        x = Reshape(shape_before_flattening)(x)

        # Layer 6 - reverse
        # x = UpSampling2D(size=(2, 2))(x)

        # Layer 5 - reverse
        x = Conv2DTranspose(filters=384, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 4 - reverse
        x = Conv2DTranspose(filters=384, kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 3 - reverse
        x = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', strides=(2, 2))(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Layer 2 - reverse
        x = Conv2DTranspose(filters=96, kernel_size=(5, 5), padding='same', strides=(2, 2))(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

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
