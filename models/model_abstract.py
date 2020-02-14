import logging
import math
import os
import sys
import traceback
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Sequence

from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.utils import plot_model
from keras_preprocessing.image import Iterator, DirectoryIterator

from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from callbacks.LossLoggingCallback import LossLoggingCallback
from utils.callbacks import step_decay_schedule, ReconstructionImagesCallback


class ModelWrapper(ABC):

    @abstractmethod
    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str):
        self.input_dim = input_dim
        self.log_dir = log_dir

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    @abstractmethod
    def compile(self, learning_rate: float, r_loss_factor: float):
        raise NotImplementedError

    def save(self):
        folder = os.path.join(self.log_dir, 'saved_model')
        os.makedirs(folder, exist_ok=True)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(os.path.join(folder, 'visualizations')):
            os.makedirs(os.path.join(folder, 'visualizations'))
        if not os.path.exists(os.path.join(folder, 'weights')):
            os.makedirs(os.path.join(folder, 'weights'))
        if not os.path.exists(os.path.join(folder, 'images')):
            os.makedirs(os.path.join(folder, 'images'))
        self.plot_model(folder)

    @abstractmethod
    def train(self, training_data, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0,
              lr_decay=1, **kwargs):
        raise NotImplementedError

    def plot_model(self, run_folder: str):
        if not hasattr(self, 'model'):
            raise AttributeError(
                "Your implementation of ModelWrapper should have an attribute model representing the whole Keras model.")
        models_to_plot = {'model': self.model}
        if hasattr(self, 'encoder'):
            models_to_plot["encoder"] = self.encoder
        if hasattr(self, 'decoder'):
            models_to_plot["decoder"] = self.encoder

        for k, v in models_to_plot.items():
            try:
                plot_model(v, os.path.join(self.log_dir, '{}.png'.format(k)))
            except Exception as e:
                logging.error("unable to save model as png")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
                    logging.error(line)
            with open(os.path.join(self.log_dir, "{}_config.json".format(k)), "w+") as f:
                f.write(self.model.to_json())

    def load_weights(self, filepath: str):
        if not hasattr(self, 'model'):
            raise AttributeError(
                "Your implementation of ModelWrapper should have an attribute model denoting the whole model.")
        self.model.load_weights(filepath)


class VAEWrapper(ModelWrapper, ABC):

    @abstractmethod
    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, kernel_visualization_layer: int, num_samples: int,
                 feature_map_layers: Sequence[int], inner_activation: str = "ReLU"):
        super().__init__(input_dim, log_dir)
        self.kernel_visualization_layer = kernel_visualization_layer
        self.num_samples = num_samples
        self.feature_map_layers = feature_map_layers
        self.inner_activation = inner_activation

    def compile(self, learning_rate, r_loss_factor):
        if not hasattr(self, 'model'):
            raise AttributeError(
                "Your implementation of VAE should have an attribute model representing the whole Keras model.")
        if not hasattr(self, 'log_var'):
            raise AttributeError(
                "Your implementation of VAE should have a layer 'log_var'.")
        if not hasattr(self, 'mu'):
            raise AttributeError(
                "Your implementation of VAE should have a layer 'mu'.")
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

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

    def save(self):
        if not hasattr(self, 'encoder') and not hasattr(self, 'decoder'):
            raise AttributeError(
                "Your VAE should have an attribute 'encoder' and an attribute 'decoder' representing a Keras model "
                "for the encoder and the decoder.")
        elif not hasattr(self, 'encoder'):
            raise AttributeError("Your VAE should have an attribute 'encoder' representing a Keras model for the "
                                 "encoder")
        elif not hasattr(self, 'decoder'):
            raise AttributeError("Your VAE should have an attribute 'decoder' representing a Keras model for the "
                                 "decoder")
        super().save()

    def train(self, training_data, batch_size, epochs, weights_folder, print_every_n_batches=100, initial_epoch=0,
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

        checkpoint_filepath = os.path.join(weights_folder, "weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(weights_folder, 'weights.h5'), save_weights_only=True, verbose=1)
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

        logging.info("Training for {} epochs".format(epochs))

        if isinstance(training_data, Iterator):
            steps_per_epoch = math.ceil(training_data.n / batch_size)
            self.model.fit_generator(
                training_data, shuffle=True, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list,
                steps_per_epoch=steps_per_epoch, workers=16
            )
        else:
            self.model.fit(
                training_data, training_data, batch_size=batch_size, shuffle=False, epochs=epochs,
                callbacks=callbacks_list
            )
        logging.info("Training finished")
