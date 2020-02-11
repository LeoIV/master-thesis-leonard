import math
import os
import sys
import traceback
from typing import Tuple, List

import logging
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Lambda, LeakyReLU, Dropout, Reshape, Conv2DTranspose, BatchNormalization, Activation
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import DirectoryIterator
from keras.utils import plot_model
from keras_preprocessing.image import Iterator

from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from callbacks.LossLoggingCallback import LossLoggingCallback
from models.AlexNet import AlexNet
import numpy as np

from utils.callbacks import step_decay_schedule, ReconstructionImagesCallback


class FrozenAlexNetVAE:
    def __init__(self, z_dim: int, use_dropout: bool, dropout_rate: float, use_batch_norm: bool,
                 shape_before_flattening: Tuple[int, int, int], input_dim: Tuple[int, int, int], log_dir: str,
                 weights_path: str,
                 kernel_visualization_layer: int, feature_map_layers: List[int], num_samples: int = 10):
        self.num_samples = num_samples
        self.feature_map_layers = feature_map_layers
        self.kernel_visualization_layer = kernel_visualization_layer
        self.weights_path = weights_path
        self.log_dir = log_dir
        self.input_dim = input_dim
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

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        # COMPILATION
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

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(os.path.join(folder, 'visualizations')):
            os.makedirs(os.path.join(folder, 'visualizations'))
        if not os.path.exists(os.path.join(folder, 'weights')):
            os.makedirs(os.path.join(folder, 'weights'))
        if not os.path.exists(os.path.join(folder, 'images')):
            os.makedirs(os.path.join(folder, 'images'))

        self.plot_model(run_folder=folder)

    def plot_model(self, run_folder):
        try:
            plot_model(self.encoder, os.path.join(self.log_dir, 'encoder.png'))
            plot_model(self.decoder, os.path.join(self.log_dir, 'decoder.png'))
            plot_model(self.model, os.path.join(self.log_dir, 'whole_model.png'))
        except Exception as e:
            logging.error("unable to save model as png")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
                logging.error(line)
        with open(os.path.join(self.log_dir, "model_config.json"), "w+") as f:
            f.write(self.model.to_json())
        with open(os.path.join(self.log_dir, "encoder_model_config.json"), "w+") as f:
            f.write(self.encoder.to_json())
        with open(os.path.join(self.log_dir, "decoder_model_config.json"), "w+") as f:
            f.write(self.decoder.to_json())

    def train(self, training_data, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0,
              lr_decay=1, embedding_samples: int = 5000):

        if isinstance(training_data, Iterator):
            training_data: DirectoryIterator
            n_batches = embedding_samples // batch_size
            if n_batches > training_data.n:
                n_batches = training_data.n
            samples = []
            for i in range(n_batches):
                samples.append(training_data.next()[0])
            embeddings_data = np.concatenate(samples, axis=0)
            training_data.reset()

        else:
            embeddings_data = training_data[:5000]

        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)
        tb_callback = TensorBoard(log_dir=self.log_dir, batch_size=batch_size, update_freq="batch")
        if self.kernel_visualization_layer >= 0:
            kv_callback = KernelVisualizationCallback(log_dir=self.log_dir, vae=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idx=self.kernel_visualization_layer)
        rc_callback = ReconstructionImagesCallback(log_dir='./logs', print_every_n_batches=print_every_n_batches,
                                                   initial_epoch=initial_epoch, vae=self)
        fm_callback = FeatureMapVisualizationCallback(log_dir=self.log_dir, model_wrapper=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idxs=self.feature_map_layers,
                                                      x_train=embeddings_data, num_samples=self.num_samples)
        ll_callback = LossLoggingCallback()
        # tb_callback has to be first as we use its filewriter subsequently but it is initialized by keras in this given order
        callbacks_list = [ll_callback, checkpoint1, checkpoint2, tb_callback, fm_callback, rc_callback, lr_sched]
        if self.kernel_visualization_layer >= 0:
            callbacks_list.append(kv_callback)

        print("Training for {} epochs".format(epochs))

        if isinstance(training_data, Iterator):
            steps_per_epoch = math.ceil(training_data.n / batch_size)
            self.model.fit_generator(
                training_data, shuffle=True, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list,
                steps_per_epoch=steps_per_epoch
            )
        else:
            self.model.fit(
                training_data, training_data, batch_size=batch_size, shuffle=False, epochs=epochs,
                callbacks=callbacks_list
            )
        print("Training finished")
