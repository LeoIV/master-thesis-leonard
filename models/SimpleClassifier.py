from typing import Sequence, Union, Tuple

from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, LeakyReLU, Dropout, ReLU, Softmax
from keras.models import Model

from models.model_abstract import DeepCNNClassifierWrapper


class SimpleClassifier(DeepCNNClassifierWrapper):

    def __init__(self, input_dim: Tuple[int, int, int], encoder_conv_filters,
                 encoder_conv_kernel_size: Sequence[Union[int, Tuple[int, int]]], num_classes: int, dropout_rate: float,
                 encoder_conv_strides: Sequence[Union[int, Tuple[int, int]]], log_dir: str, decay_rate: float,
                 feature_map_layers: Sequence[int], kernel_visualization_layer: int, num_samples: int,
                 use_batch_norm: bool = False, use_dropout: bool = False, inner_activation: str = "ReLU",
                 feature_map_reduction_factor: int = 1):

        super().__init__(input_dim=input_dim, log_dir=log_dir,
                         feature_map_reduction_factor=feature_map_reduction_factor,
                         feature_map_layers=feature_map_layers, inner_activation=inner_activation,
                         num_samples=num_samples)
        self.dropout_rate = dropout_rate
        self.kernel_visualization_layer = kernel_visualization_layer
        self.decay_rate = decay_rate
        self.num_classes = num_classes
        self.name = 'variational_autoencoder'

        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)

        self._build()

    def _build(self):

        # THE ENCODER
        model_input = Input(shape=self.input_dim, name='encoder_input')

        x = model_input

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
                x = Dropout(rate=self.dropout_rate)(x)

        x = Flatten()(x)
        x = Dense(self.num_classes)(x)
        model_output = Softmax()(x)

        self.model = Model(model_input, model_output)
        return self.model
