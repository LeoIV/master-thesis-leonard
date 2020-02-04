import logging
import os
from typing import Sequence

from PIL import Image, ImageDraw
from keras import Input, Model, losses
from keras.callbacks import Callback
from keras.layers import Conv2D, LeakyReLU
from keras.optimizers import Adam
from vis.visualization import get_num_filters, visualize_activation


class ActivationVisualizationCallback(Callback):
    def __init__(self, log_dir: str, model_wrapper: 'VariationalAutoencoder', print_every_n_batches: int,
                 layer_idxs: Sequence[int]):
        super().__init__()

        self.log_dir = log_dir
        self.model_wrapper = model_wrapper
        self.print_every_n_batches = print_every_n_batches
        self.layer_idxs = layer_idxs
        self.seen = 0


    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if batch % self.print_every_n_batches == 0 and batch > 0:
            self.seen += 1
            for i, layer_idx in enumerate(self.layer_idxs):
                # get current target layer
                layer = self.model_wrapper.encoder.layers[layer_idx]
                if layer.name not in self.model_wrapper.rfs:
                    raise AttributeError(
                        "layers of which you want to visualize the optimal stimuli have to have a defined receptive field in self.rfs")
                # use layer receptive field size as input size
                # we assume quadratic receptive fields
                logging.info("Visualizing max activations for layer {}".format(layer.name))
                input_size = max(self.model_wrapper.rfs[layer.name].rf.size[0:2])
                input_size = [input_size, input_size, self.model_wrapper.encoder.input.shape[-1].value]
                inp = x = Input(shape=input_size)
                for l in self.model_wrapper.encoder.layers[1:]:
                    if isinstance(l, Conv2D):
                        x = Conv2D(filters=l.filters, kernel_size=l.kernel_size, strides=l.strides, trainable=False)(x)
                    elif isinstance(l, LeakyReLU):
                        x = LeakyReLU(alpha=l.alpha, trainable=False)(x)
                    else:
                        raise ValueError("only Conv2D or LeakyReLU layers supported.")
                    if l.name == layer.name:
                        break
                truncated_model = Model(inp, x)
                for j, _ in enumerate(truncated_model.layers):
                    if j == 0:
                        continue
                    # copy weights for new model
                    truncated_model.layers[j].set_weights(self.model_wrapper.model.layers[j].get_weights())

                truncated_model.compile(optimizer=Adam(lr=self.model_wrapper.learning_rate),
                                        loss=losses.mean_squared_error)

                filters = list(range(get_num_filters(truncated_model.layers[layer_idx])))
                img_path = os.path.join(self.log_dir, "step_{}".format(self.seen), "max_activations",
                                        "layer_{}".format(layer_idx))
                os.makedirs(img_path, exist_ok=True)
                for f_idx in filters:
                    img = visualize_activation(truncated_model, layer_idx, filter_indices=f_idx, lp_norm_weight=0.01,
                                               tv_weight=0.01)
                    '''img = visualize_activation(self.vae.encoder, layer_idx, filter_indices=idx,
                                               seed_input=img,
                                               input_modifiers=[Jitter(0.5)])'''
                    img = Image.fromarray(img.squeeze()).resize((100, 100), resample=Image.NEAREST)
                    draw = ImageDraw.Draw(img)
                    draw.text((1, 1), "L{}F{}".format(layer_idx, f_idx), color=0)
                    img.save(os.path.join(img_path, "filter_{}.jpg".format(f_idx)))