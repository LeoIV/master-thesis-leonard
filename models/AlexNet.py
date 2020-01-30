import logging
import math
import os
from typing import Tuple, Union

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, ReLU, MaxPool2D, Softmax, BatchNormalization, \
    ZeroPadding1D
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras_preprocessing.image import DirectoryIterator, Iterator

from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from models.ModelWrapper import ModelWrapper
from utils.callbacks import step_decay_schedule


class AlexNet(ModelWrapper):

    def __init__(self, input_dim, log_dir: str,
                 use_batch_norm: bool = False,
                 use_dropout: bool = False):

        self.name = 'variational_autoencoder'

        self.input_dim = input_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.log_dir = log_dir

        self._build(self.input_dim)

    def _build(self, input_dim):

        # THE ENCODER
        input = Input(shape=self.input_dim, name='model_input')

        x = input

        # Layer 1
        x = Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(filters=256, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Layer 4
        x = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Layer 5
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Layer 6
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        # Flatten
        x = Flatten()(x)

        # FC1
        x = Dense(4096)(x)
        # , input_shape=(np.prod(self.input_dim),)
        x = ReLU()(x)
        x = Dropout(rate=0.5)(x)

        # FC2
        x = Dense(4096)(x)
        x = ReLU()(x)
        x = Dropout(rate=0.5)(x)

        # Output Layer
        x = Dense(1000)(x)
        x = Softmax()(x)

        # THE FULL VAE
        model_input = input
        model_output = x

        self.model = Model(model_input, model_output)
        logging.info("Built AlexNet model")
        return self.model

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(os.path.join(folder, 'visualizations')):
            os.makedirs(os.path.join(folder, 'visualizations'))
        if not os.path.exists(os.path.join(folder, 'weights')):
            os.makedirs(os.path.join(folder, 'weights'))
        if not os.path.exists(os.path.join(folder, 'images')):
            os.makedirs(os.path.join(folder, 'images'))

        self.plot_model(folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

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
        tb_callback = TensorBoard(log_dir=self.log_dir, batch_size=batch_size, embeddings_freq=1, update_freq="batch")
        kv_callback = KernelVisualizationCallback(log_dir=self.log_dir, vae=self,
                                                  print_every_n_batches=print_every_n_batches,
                                                  layer_idx=1)
        fm_callback = FeatureMapVisualizationCallback(log_dir=self.log_dir, model_wrapper=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idxs=[3, 7, 11, 14, 17],
                                                      x_train=embeddings_data)
        # tb_callback has to be first as we use its filewriter subsequently but it is initialized by keras in this given order
        callbacks_list = [checkpoint1, checkpoint2, tb_callback, fm_callback, kv_callback, lr_sched]

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

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'visualizations/model.png'), show_shapes=True,
                   show_layer_names=True)
