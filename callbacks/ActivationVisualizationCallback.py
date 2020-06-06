import math
import os
import time
from typing import Callable, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.callbacks import Callback
from keras.layers import Conv2D, LeakyReLU, Dense, ReLU, Activation, BatchNormalization


class ActivationVisualizationCallback(Callback):
    def __init__(self, log_dir: str, model: Model, print_every_n_batches: int, model_name: str, x_train: np.ndarray,
                 x_train_transform: Callable[[np.ndarray, Any], np.ndarray] = None,
                 transform_params: Sequence[Any] = None):
        """

        :param log_dir:
        :param model:
        :param print_every_n_batches:
        :param model_name:
        :param x_train:
        :param x_train_transform: optional function to transform x_train before passing it to the model. useful for vae decoder that needs encoded input. function uses x_train at first place and then parameters through transform_params
        """
        super().__init__()

        self.x_train_transform = x_train_transform
        self.transform_params = transform_params
        self.x_train = x_train
        self.log_dir = log_dir
        self.model = model
        self.model_name = model_name
        self.print_every_n_batches = print_every_n_batches
        self.seen = 0
        self.epoch = 1

        self._output_layers = [l for l in self.model.layers if
                               isinstance(l, (Conv2D, Dense, LeakyReLU, ReLU, Activation, BatchNormalization))]
        self._multi_output_model = Model(model.inputs, [l.output for l in self._output_layers])

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0:
            self.seen += 1

            x_train = self.x_train if self.x_train_transform is None else self.x_train_transform(self.x_train,
                                                                                                 *self.transform_params)

            outputs = self._multi_output_model.predict(x_train)

            rows = int(math.floor(math.sqrt(len(outputs))))
            cols = int(math.ceil(len(outputs) / rows))

            img_path = os.path.join(self.log_dir, "epoch_{}".format(self.epoch), "step_{}".format(self.seen),
                                    'feature_map_activations')
            os.makedirs(img_path, exist_ok=True)

            fig, axs = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10), num=round(time.time() * 10E6))

            for row in range(rows):
                for col in range(cols):
                    fig_idx = row * cols + col
                    if fig_idx >= len(outputs):
                        break
                    output = outputs[fig_idx]
                    sum_over_fms = np.mean(output, axis=(1, 2)) if len(output.shape) == 4 else output

                    if sum_over_fms.shape[1] >= 100:
                        std = np.std(sum_over_fms, axis=0)
                        mean = np.mean(sum_over_fms, axis=0)
                        axs[row][col].errorbar(list(range(len(mean))), mean, std, linestyle='None', marker='.')
                    else:
                        axs[row][col].boxplot(sum_over_fms, showfliers=False, patch_artist=True,
                                              boxprops=dict(facecolor='lightsteelblue'))
                        axs[row][col].set_xticks(np.arange(1, sum_over_fms.shape[1] + 1, 10))
                        axs[row][col].set_xticklabels(np.arange(1, sum_over_fms.shape[1] + 1, 10))
                    axs[row][col].set_title(self._output_layers[fig_idx].name)
            fig.savefig(os.path.join(img_path, '{}.png'.format(self.model_name)))
            plt.close(fig)
