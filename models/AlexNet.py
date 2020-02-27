import logging
from typing import Tuple

from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPool2D, Softmax, BatchNormalization, LeakyReLU, ReLU
from keras.models import Model

from models.model_abstract import DeepCNNClassifierWrapper


class AlexNet(DeepCNNClassifierWrapper):
    """
    An AlexNet-like image classification network.
    """

    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str,
                 use_batch_norm: bool = False,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.5, feature_map_layers=None, kernel_visualization_layer: int = -1,
                 num_samples: int = 5, use_fc: bool = True, inner_activation: str = "ReLU", decay_rate: float = 1e-7,
                 feature_map_reduction_factor: int = 1):

        super().__init__(input_dim, log_dir, feature_map_reduction_factor, inner_activation)
        self.decay_rate = decay_rate
        self.use_fc = use_fc
        if feature_map_layers is None:
            feature_map_layers = []
        self.kernel_visualization_layer = kernel_visualization_layer
        self.feature_map_layers = feature_map_layers
        self.name = 'variational_autoencoder'

        self.num_samples = num_samples

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self._build()

    def _build(self):

        # THE ENCODER
        input = Input(shape=self.input_dim, name='model_input')

        x = input

        # Layer 1
        x = Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(filters=256, kernel_size=(5, 5), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 4
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 5
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)

        # Layer 6
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        # Flatten
        x = Flatten()(x)
        if self.use_fc:
            # FC1
            x = Dense(4096)(x)
            # , input_shape=(np.prod(self.input_dim),)
            x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=self.dropout_rate)(x)

            # FC2
            x = Dense(4096)(x)
            x = LeakyReLU()(x) if self.inner_activation == "LeakyReLU" else ReLU()(x)
            if self.use_dropout:
                x = Dropout(rate=self.dropout_rate)(x)

        # Output Layer
        x = Dense(1000)(x)
        x = Softmax()(x)

        # THE FULL VAE
        model_input = input
        model_output = x

        self.model = Model(model_input, model_output)
        logging.info("Built AlexNet model")
        return self.model
