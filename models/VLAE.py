from typing import Tuple, Sequence

from keras import Input, Sequential, Model
from keras.layers import Conv2D, BatchNormalization, ReLU, SpatialDropout2D, Flatten, Dense, Concatenate, Reshape, \
    UpSampling2D, Activation, LeakyReLU, Lambda
from keras.optimizers import Adam

from models.model_abstract import VAEWrapper
from utils.vae_utils import NormalVariational, sampling

from keras import backend as K


class VLAE(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, kernel_visualization_layer: int, num_samples: int,
                 feature_map_layers: Sequence[int], inner_activation: str, decay_rate: float,
                 feature_map_reduction_factor: int, z_dim: int, dropout_rate: float = 0.05):
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dim)
        self.dropout_rate = dropout_rate
        self._build()

    def _build(self):
        ### ENCODER ###

        inputs = Input(self.input_dim)
        h_1_layers = Sequential([
            Conv2D(8, 3),
            BatchNormalization(trainable=False),
            ReLU(),
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            SpatialDropout2D(self.dropout_rate),
            ReLU()], name='h_1')
        h_1 = h_1_layers(inputs)
        h_1_flatten = Flatten()(h_1)

        h_2_layers = Sequential([
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            ReLU(),
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            SpatialDropout2D(self.dropout_rate),
            ReLU()], name='h_2')
        h_2 = h_2_layers(h_1)
        h_2_flatten = Flatten()(h_2)
        h_3_layers = Sequential([
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            ReLU(),
            Conv2D(16, 3),
            BatchNormalization(trainable=False),
            SpatialDropout2D(self.dropout_rate),
            ReLU()], name='h_3')
        h_3 = h_3_layers(h_2)
        h_3_flatten = Flatten()(h_3)

        self.mu = Dense(self.z_dim, name='mu')(h_1_flatten)
        self.log_var = Dense(self.z_dim, name='log_var')(h_1_flatten)
        z_1 = Lambda(sampling, name="z_1_latent")([self.mu, self.log_var])
        # z_1 = NormalVariational(self.z_dim, name='z_1_latent')(h_1)
        self.mu_2 = Dense(self.z_dim, name='mu_2')(h_2_flatten)
        self.log_var_2 = Dense(self.z_dim, name='log_var_2')(h_2_flatten)
        z_2 = Lambda(sampling, name="z_2_latent")([self.mu_2, self.log_var_2])
        # z_2 = NormalVariational(self.z_dim, name='z_2_latent')(h_2)
        self.mu_3 = Dense(self.z_dim, name='mu_3')(h_3_flatten)
        self.log_var_3 = Dense(self.z_dim, name='log_var_3')(h_3_flatten)
        z_3 = Lambda(sampling, name="z_3_latent")([self.mu_3, self.log_var_3])
        # z_3 = NormalVariational(self.z_dim, name='z_3_latent')(h_3)

        self.encoder = Model(inputs, [z_1, z_2, z_3], name='encoder')

        ### DECODER ###

        z_1_input, z_2_input, z_3_input = Input((self.z_dim,), name='z_1'), Input((self.z_dim,), name='z_2'), Input(
            (self.z_dim,), name='z_3')

        z_3 = Dense(1024, activation='relu')(z_3_input)
        z_tilde_3_layers = Sequential([
            Dense(1024),
            BatchNormalization(trainable=False),
            ReLU(), Dense(1024),
            BatchNormalization(trainable=False),
            ReLU(), Dense(1024),
            BatchNormalization(trainable=False),
            ReLU()], name='z_tilde_3')
        z_tilde_3 = z_tilde_3_layers(z_3)

        z_2 = Dense(128, activation='relu')(z_2_input)
        z_tilde_2_layers = Sequential([
            Dense(128),
            BatchNormalization(trainable=False),
            ReLU(), Dense(128),
            BatchNormalization(trainable=False),
            ReLU(), Dense(128),
            BatchNormalization(trainable=False),
            ReLU()], name='z_tilde_2')
        input_z_tilde_2 = Concatenate()([z_tilde_3, z_2])
        z_tilde_2 = z_tilde_2_layers(input_z_tilde_2)

        z_1 = Dense(128, activation='relu')(z_1_input)
        z_tilde_1_layers = Sequential([
            Dense(128),
            BatchNormalization(trainable=False),
            ReLU(), Dense(128),
            BatchNormalization(trainable=False),
            ReLU(), Dense(128),
            BatchNormalization(trainable=False),
            ReLU()], name='z_tilde_1')
        input_z_tilde_1 = Concatenate()([z_tilde_2, z_1])
        z_tilde_1 = z_tilde_1_layers(input_z_tilde_1)

        decoder = Reshape((2, 2, 32))(z_tilde_1)
        decoder = UpSampling2D(2)(decoder)  # 4x4
        decoder = Conv2D(32, 3)(decoder)  # 2x2
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = ReLU()(decoder)
        decoder = UpSampling2D(4)(decoder)  # 8x8
        decoder = Conv2D(16, 3)(decoder)  # 6x6
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = ReLU()(decoder)
        decoder = UpSampling2D(2)(decoder)  # 12x12
        decoder = Conv2D(8, 3)(decoder)  # 10x10
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = ReLU()(decoder)
        decoder = UpSampling2D(2)(decoder)  # 20x20
        decoder = Conv2D(4, 5)(decoder)  # 16x16
        decoder = BatchNormalization(trainable=False)(decoder)
        decoder = LeakyReLU()(decoder)
        decoder = UpSampling2D(2)(decoder)  # 32x32
        decoder = Conv2D(1, 5)(decoder)  # 28x28
        decoder = Activation('sigmoid')(decoder)

        self.decoder = Model([z_1_input, z_2_input, z_3_input], decoder, name='decoder')

        # def _make_vlae(latent_size):

        z_1, z_2, z_3 = self.encoder(inputs)
        decoded = self.decoder([z_1, z_2, z_3])
        self.model = Model(inputs, decoded, name='vlae')
        return self.encoder

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

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

        optimizer = Adam(lr=learning_rate, decay=self.decay_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])
