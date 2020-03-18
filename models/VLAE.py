import math
from typing import Tuple, Sequence, List

from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate, Reshape, \
    Activation, Lambda, Conv2DTranspose
from keras.optimizers import Adam

from models.model_abstract import VAEWrapper
from utils.vae_utils import sampling


class VLAE(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, kernel_visualization_layer: int, num_samples: int,
                 feature_map_layers: Sequence[int], inner_activation: str, decay_rate: float,
                 feature_map_reduction_factor: int, z_dims: List[int], dropout_rate: float = 0.05):
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dims, ["mu_1", "mu_2", "mu_3"],
                         ["log_var_1", "log_var_2", "log_var_3"])
        self.dropout_rate = dropout_rate
        self._build()

    def _build(self):
        inputs = Input(self.input_dim)

        # INFERENCE 0
        i0 = Conv2D(filters=math.ceil(64 / self.feature_map_reduction_factor), kernel_size=[4, 4], strides=2)(inputs)
        i0 = BatchNormalization()(i0)
        i0 = ReLU()(i0)
        i0 = Conv2D(filters=math.ceil(128 / self.feature_map_reduction_factor), kernel_size=[4, 4], strides=2)(i0)
        i0 = BatchNormalization()(i0)
        i0 = ReLU()(i0)

        # LADDER 0
        l0 = Conv2D(filters=math.ceil(64 / self.feature_map_reduction_factor), kernel_size=4, strides=2)(inputs)
        l0 = BatchNormalization()(l0)
        l0 = ReLU()(l0)
        l0 = Flatten()(l0)
        self.mu_0 = Dense(self.z_dims[0], name='mu_1')(l0)
        self.log_var_0 = Dense(self.z_dims[0], name='log_var_1')(l0)
        z_0 = Lambda(sampling, name="z_1_latent")([self.mu_0, self.log_var_0])

        # INFERENCE 1
        i1 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(i0)
        i1 = BatchNormalization()(i1)
        i1 = ReLU()(i1)
        i1 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(i1)
        i1 = BatchNormalization()(i1)
        i1 = ReLU()(i1)

        # LADDER 1
        l1 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(i0)
        l1 = BatchNormalization()(l1)
        l1 = ReLU()(l1)
        l1 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(l1)
        l1 = BatchNormalization()(l1)
        l1 = ReLU()(l1)
        l1 = Flatten()(l1)
        self.mu_1 = Dense(self.z_dims[1], name='mu_2')(l1)
        self.log_var_1 = Dense(self.z_dims[1], name='log_var_2')(l1)
        z_1 = Lambda(sampling, name="z_2_latent")([self.mu_1, self.log_var_1])

        # LADDER 2
        l2 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(i1)
        l2 = BatchNormalization()(l2)
        l2 = ReLU()(l2)
        l2 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(l2)
        l2 = BatchNormalization()(l2)
        l2 = ReLU()(l2)
        l2 = Flatten()(l2)
        self.mu_2 = Dense(self.z_dims[2], name='mu_3')(l2)
        self.log_var_2 = Dense(self.z_dims[2], name='log_var_3')(l2)
        z_2 = Lambda(sampling, name="z_3_latent")([self.mu_2, self.log_var_2])

        encoder_output = [z_0, z_1, z_2]

        self.encoder = Model(inputs, encoder_output, name='encoder')

        ### DECODER ###

        z_1_input, z_2_input, z_3_input = Input((self.z_dims[0],), name='z_1'), Input((self.z_dims[1],),
                                                                                      name='z_2'), Input(
            (self.z_dims[2],), name='z_3')

        # GENERATIVE 2
        g2 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(z_3_input)
        g2 = BatchNormalization()(g2)
        g2 = ReLU()(g2)
        g2 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(g2)
        g2 = BatchNormalization()(g2)
        g2 = ReLU()(g2)
        g2 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(g2)

        # GENERATIVE 1
        g1 = Concatenate()([g2, z_2_input])
        g1 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(g1)
        g1 = BatchNormalization()(g1)
        g1 = ReLU()(g1)
        g1 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(g1)
        g1 = BatchNormalization()(g1)
        g1 = ReLU()(g1)
        g1 = Dense(math.ceil(1024 / self.feature_map_reduction_factor))(g1)

        # GENERATIVE 0
        g0 = Concatenate()([g1, z_1_input])
        g0 = Dense(6 * 6 * 128)(g0)  # TODO make dynamic
        g0 = BatchNormalization()(g0)
        g0 = ReLU()(g0)
        g0 = Reshape((6, 6, 128))(g0)  # TODO make dynamic
        g0 = Conv2DTranspose(filters=math.ceil(64 / self.feature_map_reduction_factor), kernel_size=3, strides=2)(g0)
        g0 = BatchNormalization()(g0)
        g0 = ReLU()(g0)
        g0 = Conv2DTranspose(filters=self.input_dim[-1], kernel_size=4, strides=2)(g0)
        g0 = Activation('sigmoid')(g0)

        self.decoder = Model([z_1_input, z_2_input, z_3_input], g0, name='decoder')
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
