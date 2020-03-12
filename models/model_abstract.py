import logging
import math
import os
import sys
import traceback
from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Optional, Union

import numpy as np
from keras import backend as K
from keras.backend import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.utils import plot_model
from keras_preprocessing.image import Iterator, DirectoryIterator

from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.HiddenSpaceCallback import HiddenSpaceCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from callbacks.LossLoggingCallback import LossLoggingCallback
from callbacks.ReconstructionCallback import ReconstructionImagesCallback
from utils.callbacks import step_decay_schedule
from utils.set_utils import has_intersection


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
    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0,
              y_train: Optional[np.ndarray] = None, x_test: Optional[Union[Iterator, np.ndarray]] = None,
              y_test: Optional[np.ndarray] = None, lr_decay=1, training_labels: Optional[np.ndarray] = None, **kwargs):
        raise NotImplementedError

    def plot_model(self, run_folder: str):
        if not hasattr(self, 'model'):
            raise AttributeError(
                "Your implementation of ModelWrapper should have an attribute model representing the whole Keras model.")
        models_to_plot = {'model': self.model}
        if hasattr(self, 'encoder'):
            models_to_plot["encoder"] = self.encoder
        if hasattr(self, 'decoder'):
            models_to_plot["decoder"] = self.decoder

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


class DeepCNNModelWrapper(ModelWrapper, ABC):

    @abstractmethod
    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, feature_map_reduction_factor: int,
                 inner_activation: str):
        super().__init__(input_dim, log_dir)
        self.feature_map_reduction_factor = feature_map_reduction_factor
        self.inner_activation = inner_activation


class DeepCNNClassifierWrapper(DeepCNNModelWrapper, ABC):

    @abstractmethod
    def __init__(self, feature_map_layers: Sequence[int], input_dim: Tuple[int, int, int], log_dir: str,
                 feature_map_reduction_factor: int, inner_activation: str, num_samples: int):

        super().__init__(input_dim, log_dir, feature_map_reduction_factor, inner_activation)
        self.num_samples = num_samples
        self.feature_map_layers = feature_map_layers

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        optimizer = Adam(lr=learning_rate, decay=self.decay_rate)
        self.model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

    def train(self, x_train, batch_size, epochs, weights_folder, print_every_n_batches=100, initial_epoch=0,
              lr_decay=1, embedding_samples: int = 5000, y_train: Optional[np.ndarray] = None,
              x_test: Optional[Union[Iterator, np.ndarray]] = None, y_test: Optional[np.ndarray] = None,
              steps_per_epoch: int = None):

        if isinstance(x_train, Iterator):
            x_train: DirectoryIterator
            n_batches = embedding_samples // batch_size
            if n_batches > x_train.n:
                n_batches = x_train.n
            samples = []
            for i in range(n_batches):
                samples.append(x_train.next()[0])
            embeddings_data = np.concatenate(samples, axis=0)
            x_train.reset()

        else:
            embeddings_data = x_train[:5000]

        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(weights_folder, "weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(weights_folder, 'weights.h5'), save_weights_only=True,
                                      verbose=1)
        tb_callback = TensorBoard(log_dir=self.log_dir, batch_size=batch_size, update_freq="batch")
        if self.kernel_visualization_layer >= 0:
            kv_callback = KernelVisualizationCallback(log_dir=self.log_dir, vae=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idx=self.kernel_visualization_layer)
        fm_callback = FeatureMapVisualizationCallback(log_dir=self.log_dir, model_wrapper=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idxs=self.feature_map_layers,
                                                      x_train=embeddings_data, num_samples=self.num_samples)
        ll_callback = LossLoggingCallback(self.log_dir)
        # tb_callback has to be first as we use its filewriter subsequently but it is initialized by keras in this given order
        callbacks_list = [ll_callback, checkpoint1, checkpoint2, tb_callback, fm_callback, lr_sched]
        if self.kernel_visualization_layer >= 0:
            callbacks_list.append(kv_callback)

        print("Training for {} epochs".format(epochs))

        if isinstance(x_train, Iterator):
            steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else math.ceil(x_train / batch_size)
            self.model.fit_generator(
                generator=x_train, shuffle=True, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list,
                steps_per_epoch=steps_per_epoch, workers=16, validation_data=x_test
            )
        else:
            self.model.fit(
                x=x_train, y=y_train, batch_size=batch_size if steps_per_epoch is None else None,
                shuffle=True, epochs=epochs,
                steps_per_epoch=steps_per_epoch if steps_per_epoch is not None else None,
                callbacks=callbacks_list, validation_data=(x_test, y_test),
                validation_steps=steps_per_epoch if steps_per_epoch is not None else None,
            )
        print("Training finished")


class VAEWrapper(DeepCNNModelWrapper, ABC):

    @abstractmethod
    def __init__(self, input_dim: Tuple[int, int, int], log_dir: str, kernel_visualization_layer: int, num_samples: int,
                 feature_map_layers: Sequence[int], inner_activation: str, decay_rate: float,
                 feature_map_reduction_factor: int, z_dims: Sequence[int], mu_layer_names: Sequence[str],
                 logvar_layer_names: Sequence[str]):
        super().__init__(input_dim, log_dir, feature_map_reduction_factor, inner_activation)

        self.logvar_layer_names = logvar_layer_names
        self.mu_layer_names = mu_layer_names
        self.z_dims = z_dims
        self.decay_rate = decay_rate
        self.kernel_visualization_layer = kernel_visualization_layer
        self.num_samples = num_samples
        self.feature_map_layers = feature_map_layers
        self.mu_layers = []
        self.logvar_layers = []
        if not (len(self.z_dims) == len(self.mu_layer_names) == len(self.logvar_layer_names)):
            raise RuntimeError("Length of z_dims, mu_layer_names, and logvar_layer_names must be equal.")

    def compile(self, learning_rate, r_loss_factor):
        if not hasattr(self, 'model'):
            raise AttributeError(
                "Your implementation of VAE should have an attribute model representing the whole Keras model.")
        for layer in self.model.layers:
            for mu_layer_name in self.mu_layer_names:
                if layer.name == mu_layer_name:
                    self.mu_layers.append(layer)
            for logvar_layer_name in self.logvar_layer_names:
                if layer.name == logvar_layer_name:
                    self.logvar_layers.append(layer)
        if has_intersection(self.mu_layers, self.logvar_layers):
            raise RuntimeError(
                "There is an intersection between mu_layers and logvar_layers. That should not occur. Check your naming.")
        if len(self.mu_layers) != len(self.mu_layer_names):
            raise RuntimeError("Did not find all mu layers given by mu_layer_names")
        if len(self.logvar_layers) != len(self.logvar_layer_names):
            raise RuntimeError("Did not find all logvar layers given by logvar_layer_names")

        self.learning_rate = learning_rate

        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = 0.0
            for lv, m in zip(self.logvar_layers, self.mu_layers):
                kl_loss += -0.5 * K.sum(1 + lv.output - K.square(m.output) - K.exp(lv.output), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = Adam(lr=learning_rate, decay=self.decay_rate)
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

    def train(self, x_train: Union[Iterator, np.ndarray], batch_size, epochs, weights_folder, print_every_n_batches=100,
              initial_epoch=0, lr_decay=1, embedding_samples: int = 5000, y_train: Optional[np.ndarray] = None,
              x_test: Optional[Union[Iterator, np.ndarray]] = None, y_test: Optional[np.ndarray] = None,
              steps_per_epoch: int = None):

        if isinstance(x_train, Iterator):
            x_train: DirectoryIterator
            n_batches = embedding_samples // batch_size
            if n_batches > x_train.n:
                n_batches = x_train.n
            samples = []
            for i in range(n_batches):
                samples.append(x_train.next()[0])
            x_train_subset = np.concatenate(samples, axis=0)
            x_train.reset()
            y_train = None
            y_embedding = None

        else:
            x_train_subset = x_train[:5000]
            y_embedding = y_train[:5000] if y_train is not None else None

        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(weights_folder, "weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(weights_folder, 'weights.h5'), save_weights_only=True, verbose=1)
        tb_callback = TensorBoard(log_dir=self.log_dir, batch_size=batch_size, update_freq="batch")
        if self.kernel_visualization_layer >= 0:
            kv_callback = KernelVisualizationCallback(log_dir=self.log_dir, print_every_n_batches=print_every_n_batches,
                                                      layer_idx=self.kernel_visualization_layer)
        rc_callback = ReconstructionImagesCallback(log_dir=self.log_dir, print_every_n_batches=print_every_n_batches,
                                                   initial_epoch=initial_epoch, vae=self, x_train=x_train_subset,
                                                   num_reconstructions=self.num_samples, num_inputs=len(self.z_dims))
        fm_callback = FeatureMapVisualizationCallback(log_dir=self.log_dir, model_wrapper=self,
                                                      print_every_n_batches=print_every_n_batches,
                                                      layer_idxs=self.feature_map_layers,
                                                      x_train=x_train_subset, num_samples=self.num_samples)
        hs_callback = HiddenSpaceCallback(log_dir=self.log_dir, vae=self, batch_size=batch_size,
                                          x_train=x_train_subset, y_train=y_embedding, max_samples=5000,
                                          layer_names=self.mu_layer_names)
        ll_callback = LossLoggingCallback(logdir=self.log_dir)
        # tb_callback has to be first as we use its filewriter subsequently but it is initialized by keras in this given order
        callbacks_list = [hs_callback, ll_callback, checkpoint1, checkpoint2, tb_callback, fm_callback, rc_callback,
                          lr_sched]
        if self.kernel_visualization_layer >= 0:
            callbacks_list.append(kv_callback)

        logging.info("Training for {} epochs".format(epochs))

        if isinstance(x_train, Iterator):
            steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else math.ceil(x_train.n / batch_size)
            self.model.fit_generator(
                generator=x_train, shuffle=True, epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list,
                steps_per_epoch=steps_per_epoch, workers=16, validation_data=x_test
            )
        else:
            self.model.fit(
                x=x_train, y=x_train, batch_size=batch_size if steps_per_epoch is None else None,
                shuffle=True, epochs=epochs,
                steps_per_epoch=steps_per_epoch if steps_per_epoch is not None else None,
                callbacks=callbacks_list, validation_data=(x_test, x_test),
                validation_steps=steps_per_epoch if steps_per_epoch is not None else None,
            )
        logging.info("Training finished")
