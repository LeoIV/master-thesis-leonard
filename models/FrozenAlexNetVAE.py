import logging
import math
import os
import sys
import traceback
from typing import Tuple, List

import numpy as np
from keras import Model, Input
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Lambda, LeakyReLU, Dropout, Reshape, Conv2DTranspose, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import DirectoryIterator
from keras_preprocessing.image import Iterator

from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from callbacks.LossLoggingCallback import LossLoggingCallback
from models.AlexNet import AlexNet
from models.model_abstract import ModelWrapper, VAEWrapper
from utils.callbacks import step_decay_schedule, ReconstructionImagesCallback


class FrozenAlexNetVAE(VAEWrapper):
    def __init__(self, z_dim: int, use_dropout: bool, dropout_rate: float, use_batch_norm: bool,
                 shape_before_flattening: Tuple[int, int, int], input_dim: Tuple[int, int, int], log_dir: str,
                 weights_path: str, kernel_visualization_layer: int, feature_map_layers: List[int],
                 num_samples: int = 10):
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers)
        self.weights_path = weights_path
        self.shape_before_flattening = shape_before_flattening
        self.z_dim = z_dim
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self._build()

        # TODO remove
        '''    
        z_dim = 2000
        use_dropout = False
        dropout_rate = 0.3
        use_batch_norm = True
        shape_before_flattening = (7, 7, 256)
        input_dim = (224, 224, 3)
        '''

    def _build(self):

        model = AlexNet(input_dim=(224, 224, 3), use_batch_norm=self.use_batch_norm,
                        use_dropout=self.use_dropout, dropout_rate=self.dropout_rate, log_dir=self.log_dir)
        model = model.model
        try:
            model.load_weights(self.weights_path)
        except Exception as e:
            logging.error(
                "Failed while restoring AlexNet. Make sure you set the configuration accordingly to the AlexNet you're "
                "attempting to load.")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
                logging.error(line)
            raise e
        logging.info("Successfully restored AlexNet weights")
        for layer in model.layers:
            layer.trainable = False
            logging.debug("Setting {} to non-trainable".format(layer.name))
        model_input = model.inputs
        # chop of last fc layer (uncluding softmax and leaky relu)
        last_layer = model.layers[-4].output
        self.mu = Dense(self.z_dim, name='mu')(last_layer)
        self.log_var = Dense(self.z_dim, name='log_var')(last_layer)
        self.encoder_mu_log_var = Model(model_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(model_input, encoder_output)

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = decoder_input

        # FC2 - reverse
        x = Dense(4096)(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)
        # FC1 - reverse
        x = Dense(np.prod(self.shape_before_flattening))(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)

        # Unflatten
        x = Reshape(self.shape_before_flattening)(x)

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
        model_input = model_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)