from typing import Tuple, Sequence, Optional, Union, Iterator

from keras import Input, Sequential, Model
from keras.layers import Conv2D, BatchNormalization, ReLU, SpatialDropout2D, Flatten, Dense, Concatenate, Reshape, \
    UpSampling2D, Activation, LeakyReLU, Lambda, Conv2DTranspose
from keras.optimizers import Adam

from models.model_abstract import VAEWrapper
from utils.vae_utils import NormalVariational, sampling

from keras import backend as K

import numpy as np


class HLAE(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, kernel_visualization_layer: int, num_samples: int,
                 feature_map_layers: Sequence[int], inner_activation: str, decay_rate: float,
                 feature_map_reduction_factor: int, z_dims: Sequence[int], dropout_rate: float = 0.05):
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dims, ["mu_1", "mu_2", "mu_3"],
                         ["log_var_1", "log_var_2", "log_var_3"])
        self.dropout_rate = dropout_rate
        self._build()

    def _build(self):
        inputs = Input(self.input_dim)

        # LADDER 0
        l0 = Conv2D(filters=64, kernel_size=4, strides=2)(inputs)
        l0 = BatchNormalization()(l0)
        l0 = ReLU()(l0)
        l0 = Conv2D(filters=64, kernel_size=4, strides=2)(l0)
        l0 = BatchNormalization()(l0)
        l0 = ReLU()(l0)
        l0 = Flatten()(l0)
        mu_0 = Dense(self.z_dims[0], name='mu_1')(l0)
        log_var_0 = Dense(self.z_dims[0], name='log_var_1')(l0)
        z_0 = Lambda(sampling, name="z_1_latent")([mu_0, log_var_0])

        # LADDER 1
        l1 = Dense(1024)(z_0)
        l1 = BatchNormalization()(l1)
        l1 = ReLU()(l1)
        l1 = Dense(1024)(l1)
        l1 = BatchNormalization()(l1)
        l1 = ReLU()(l1)
        l1 = Flatten()(l1)
        mu_1 = Dense(self.z_dims[1], name='mu_2')(l1)
        log_var_1 = Dense(self.z_dims[1], name='log_var_2')(l1)
        z_1 = Lambda(sampling, name="z_2_latent")([mu_1, log_var_1])

        # LADDER 2
        l2 = Dense(1024)(z_1)
        l2 = BatchNormalization()(l2)
        l2 = ReLU()(l2)
        l2 = Dense(1024)(l2)
        l2 = BatchNormalization()(l2)
        l2 = ReLU()(l2)
        l2 = Flatten()(l2)
        mu_2 = Dense(self.z_dims[2], name='mu_3')(l2)
        log_var_2 = Dense(self.z_dims[2], name='log_var_3')(l2)
        z_2 = Lambda(sampling, name="z_3_latent")([mu_2, log_var_2])

        encoder_output = [z_0, z_1, z_2]

        self.encoder = Model(inputs, encoder_output, name='encoder')

        ### DECODER ###

        z_1_input, z_2_input, z_3_input = Input((self.z_dims[0],), name='z_1'), Input((self.z_dims[1],),
                                                                                      name='z_2'), Input(
            (self.z_dims[2],), name='z_3')

        # GENERATIVE 2
        g2 = Dense(1024)(z_3_input)
        g2 = BatchNormalization()(g2)
        g2 = ReLU()(g2)
        g2 = Dense(1024)(g2)
        g2 = BatchNormalization()(g2)
        g2 = ReLU()(g2)
        g2 = Dense(1024)(g2)

        # GENERATIVE 1
        g1 = Concatenate()([g2, z_2_input])
        g1 = Dense(1024)(g1)
        g1 = BatchNormalization()(g1)
        g1 = ReLU()(g1)
        g1 = Dense(1024)(g1)
        g1 = BatchNormalization()(g1)
        g1 = ReLU()(g1)
        g1 = Dense(1024)(g1)

        # GENERATIVE 0
        g0 = Concatenate()([g1, z_1_input])
        g0 = Dense(6 * 6 * 128)(g0)  # TODO make dynamic
        g0 = BatchNormalization()(g0)
        g0 = ReLU()(g0)
        g0 = Reshape((6, 6, 128))(g0)  # TODO make dynamic
        g0 = Conv2DTranspose(filters=64, kernel_size=3, strides=2)(g0)
        g0 = BatchNormalization()(g0)
        g0 = ReLU()(g0)
        g0 = Conv2DTranspose(filters=self.input_dim[-1], kernel_size=4, strides=2)(g0)
        g0 = Activation('sigmoid')(g0)

        self.decoder = Model([z_1_input, z_2_input, z_3_input], g0, name='decoder')
        decoder_output = self.decoder(encoder_output)

        self.model = Model(inputs, decoder_output, name='vlae')
        return self.encoder

    def train(self, x_train: Union[Iterator, np.ndarray], batch_size, epochs, weights_folder, print_every_n_batches=100,
              initial_epoch: int = 0, lr_decay: float = 1, embedding_samples: int = 5000,
              y_train: Optional[np.ndarray] = None,
              x_test: Optional[Union[Iterator, np.ndarray]] = None, y_test: Optional[np.ndarray] = None,
              steps_per_epoch: int = None):
        super().train(x_train, batch_size, epochs, weights_folder, print_every_n_batches, initial_epoch, lr_decay,
                      embedding_samples, y_train, x_test, y_test, steps_per_epoch)
