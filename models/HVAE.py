import math
from typing import Tuple, Sequence, Optional, Union, Iterator

import numpy as np
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Reshape, \
    Activation, Lambda, Conv2DTranspose, Input

from models.model_abstract import VAEWrapper
from utils.vae_utils import sampling


class HVAE(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, kernel_visualization_layer: int, num_samples: int,
                 feature_map_layers: Sequence[int], inner_activation: str, decay_rate: float,
                 feature_map_reduction_factor: int, z_dims: Sequence[int], dropout_rate: float = 0.05):
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dims,
                         ["mu_{}".format(i) for i in range(1, 6)],
                         ["log_var_{}".format(i) for i in range(1, 6)])
        self.dropout_rate = dropout_rate
        self._build()

    def _build(self):
        encoder_input = Input(self.input_dim)

        # LADDER 0
        x = Conv2D(filters=math.ceil(64 / self.feature_map_reduction_factor), kernel_size=4, strides=2)(encoder_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Flatten()(x)
        mu_0 = Dense(self.z_dims[0], name='mu_1')(x)
        log_var_0 = Dense(self.z_dims[0], name='log_var_1')(x)
        x = Lambda(sampling, name="z_1_latent")([mu_0, log_var_0])

        # LADDER 1
        x = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        mu_1 = Dense(self.z_dims[1], name='mu_2')(x)
        log_var_1 = Dense(self.z_dims[1], name='log_var_2')(x)
        x = Lambda(sampling, name="z_2_latent")([mu_1, log_var_1])

        # LADDER 2
        x = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        mu_2 = Dense(self.z_dims[2], name='mu_3')(x)
        log_var_2 = Dense(self.z_dims[2], name='log_var_3')(x)
        encoder_output = Lambda(sampling, name="z_3_latent")([mu_2, log_var_2])

        self.encoder = Model(encoder_input, encoder_output)

        ### DECODER ###

        decoder_input = Input((self.z_dims[2],), name='decoder_input')

        # GENERATIVE 2
        x = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(decoder_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(x)

        mu_4 = Dense(self.z_dims[1], name='mu_4')(x)
        log_var_4 = Dense(self.z_dims[1], name='log_var_4')(x)
        x = Lambda(sampling, name="z_4_latent")([mu_4, log_var_4])

        # GENERATIVE 1
        x = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        mu_5 = Dense(self.z_dims[0], name='mu_5')(x)
        log_var_5 = Dense(self.z_dims[0], name='log_var_5')(x)
        x = Lambda(sampling, name="z_5_latent")([mu_5, log_var_5])

        # GENERATIVE 0
        x = Dense(6 * 6 * 128)(x)  # TODO make dynamic
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Reshape((6, 6, 128))(x)  # TODO make dynamic
        x = Conv2DTranspose(filters=(64 / self.feature_map_reduction_factor), kernel_size=3, strides=2)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters=self.input_dim[-1], kernel_size=4, strides=2)(x)
        decoder_output = Activation('sigmoid')(x)

        self.decoder = Model(decoder_input, decoder_output)
        model_output = self.decoder(encoder_output)

        self.model = Model(encoder_input, model_output)
        return self.encoder
