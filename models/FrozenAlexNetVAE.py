import logging
import math
import sys
import traceback
from typing import Tuple, List, Sequence

import numpy as np
from keras import Model, Input
from keras.layers import Dense, Lambda, LeakyReLU, Dropout, Reshape, Conv2DTranspose, BatchNormalization, Activation, \
    ReLU

from models.AlexNet import AlexNet
from models.model_abstract import VAEWrapper
from utils.vae_utils import sampling


class FrozenAlexNetVAE(VAEWrapper):
    def __init__(self, z_dims: Sequence[int], use_dropout: bool, dropout_rate: float, use_batch_norm: bool,
                 shape_before_flattening: Tuple[int, int, int], input_dim: Tuple[int, int, int], log_dir: str,
                 weights_path: str, kernel_visualization_layer: int, feature_map_layers: List[int],
                 num_samples: int = 10, inner_activation: str = "ReLU", decay_rate: float = 1e-7,
                 feature_map_reduction_factor: int = 1):
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dims, ["mu"], ["log_var"])

        if len(self.z_dims) > 1:
            raise RuntimeError("Only one z_dim allowed for this model")
        self.weights_path = weights_path
        self.shape_before_flattening = shape_before_flattening
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
        mu = Dense(self.z_dims[0], name='mu')(last_layer)
        log_var = Dense(self.z_dims[0], name='log_var')(last_layer)

        encoder_output = Lambda(sampling, name='encoder_output')([mu, log_var])

        self.encoder = Model(model_input, encoder_output)

        decoder_input = Input(shape=(self.z_dims[0],), name='decoder_input')

        x = decoder_input

        # FC2 - reverse
        x = Dense(math.ceil(4096 / self.feature_map_reduction_factor))(x)
        x = LeakyReLU()(x)
        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)
        # FC1 - reverse
        x = Dense(np.prod(self.shape_before_flattening))(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        if self.use_dropout:
            x = Dropout(rate=self.dropout_rate)(x)

        # Unflatten
        x = Reshape(self.shape_before_flattening)(x)

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
        model_input = model_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)
