import math
from typing import Tuple, Sequence, List

from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate, Reshape, \
    Activation, Lambda, Conv2DTranspose, Dropout, ELU
from keras.optimizers import Adam

from models.model_abstract import VAEWrapper
from utils.vae_utils import sampling, precision_weighted_sampler

import numpy as np


class LVAE(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int],
                 use_dropout: bool,
                 use_batch_norm: bool,
                 log_dir: str,
                 kernel_visualization_layer: int,
                 num_samples: int,
                 feature_map_layers: Sequence[int],
                 inner_activation: str,
                 decay_rate: float,
                 feature_map_reduction_factor: int,
                 dropout_rate: float = 0.3,
                 z_dims: List[int] = [64, 32, 32, 32, 32],
                 mlp_sizes: List[int] = [512, 256, 256, 128, 128]):
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dims, ["mu_1", "mu_2", "mu_3"],
                         ["log_var_1", "log_var_2", "log_var_3"])
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.mlp_sizes = mlp_sizes
        self.l = len(self.mlp_sizes)

        self.e_mus, self.e_logvars = [0] * self.l, [0] * self.l
        self.p_mus, self.p_logvars = [0] * self.l, [0] * self.l
        self.d_mus, self.d_logvars = [0] * self.l, [0] * self.l

        self._build()

    def _build(self):

        def vae_sampler(x, size, activation="elu"):
            mu = Dense(size)(x)
            log_var = Dense(size)(x)
            return sampling([mu, log_var]), mu, log_var

        inputs = Input(self.input_dim)

        ### ENCODER ###

        h = Flatten()(inputs)
        for l in range(self.l):
            h = Dense(self.mlp_sizes[l])(h)
            h = BatchNormalization()(h)
            h = ELU()(h)
            _, self.e_mus[l], self.e_logvars[l] = vae_sampler(h, self.z_dims[l], activation='softplus')

        self.encoder = Model(inputs, self.e_mus, name='encoder')

        ### DECODER ###

        for l in range(self.l - 1, -1, -1):
            if l == self.l - 1:
                mu, logvar = self.e_mus[l], self.e_logvars[l]
                self.d_mus[l], self.d_logvars[l] = mu, logvar
                z = sampling([self.d_mus[l], self.d_logvars[l]])
                # prior of z_L is set as standard Gaussian, N(0,I).
                self.p_mus[l], self.p_logvars[l] = K.zeros(K.int_shape(mu)), K.zeros(K.int_shape(logvar))
            else:
                # prior is developed from z of the above layer
                _, self.p_mus[l], self.p_logvars[l] = vae_sampler(z, self.z_dims[l], 'softplus')
                z, self.d_mus[l], self.d_logvars[l] = precision_weighted_sampler(
                    (self.e_mus[l], K.exp(self.e_logvars[l])), (self.p_mus[l], K.exp(self.p_logvars[l])))
        x = Dense(np.prod(self.input_dim))(z)
        x = ELU(x)
        x = Reshape(self.input_dim)(x)

        self.decoder = Model([input], g0, name='decoder')
        decoder_output = self.decoder(encoder_output)

        self.model = Model(inputs, decoder_output, name='vlae')
        return self.encoder

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = 0.0
            for lv, m in zip([self.log_var_0, self.log_var_1, self.log_var_2], [self.mu_0, self.mu_1, self.mu_2]):
                kl_loss += -0.5 * K.sum(1 + lv - K.square(m) - K.exp(lv), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = Adam(lr=learning_rate, decay=self.decay_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])
