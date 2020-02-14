from typing import Sequence, Union, Tuple

import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout, ReLU
from keras.models import Model
from receptivefield.keras import KerasReceptiveField

from models.model_abstract import VAEWrapper


class VariationalAutoencoder(VAEWrapper):
    def __init__(self, input_dim: Tuple[int, int, int], encoder_conv_filters,
                 encoder_conv_kernel_size: Sequence[Union[int, Tuple[int, int]]],
                 encoder_conv_strides: Sequence[Union[int, Tuple[int, int]]], decoder_conv_t_filters,
                 decoder_conv_t_kernel_size: Sequence[Union[int, Tuple[int, int]]],
                 decoder_conv_t_strides: Sequence[Union[int, Tuple[int, int]]], z_dim: int, log_dir: str,
                 feature_map_visualization_layers: Sequence[int], kernel_visualization_layer: int,
                 use_batch_norm: bool = False, use_dropout: bool = False, num_samples: int = 10,
                 inner_activation: str = "ReLU"):

        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_visualization_layers,
                         inner_activation)
        self.name = 'variational_autoencoder'

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

        layer_names = ["encoder_conv_{}".format(i) for i in range(self.n_layers_encoder)]
        self.rfs = KerasReceptiveField(self._build).compute(input_shape=self.input_dim[0:2],
                                                            input_layer="encoder_input",
                                                            output_layers=layer_names)
        self.rfs = dict(map(lambda x: (x[0], x[1]), zip(layer_names, self.rfs)))

    def _build(self):

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

            x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

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
