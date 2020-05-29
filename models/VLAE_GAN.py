import logging
import math
import os
import time
from typing import Tuple, Sequence, List, Union, Optional

import numpy as np
from keras import Input, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate, Reshape, \
    Activation, Lambda, Conv2DTranspose, Dropout, LeakyReLU, MaxPool2D, UpSampling2D
from keras.optimizers import Adam
from keras_preprocessing.image import Iterator, DirectoryIterator
from tqdm import tqdm

from callbacks.ActivationVisualizationCallback import ActivationVisualizationCallback
from callbacks.FeatureMapVisualizationCallback import FeatureMapVisualizationCallback
from callbacks.HiddenSpaceCallback import HiddenSpaceCallback
from callbacks.KernelVisualizationCallback import KernelVisualizationCallback
from callbacks.LossLoggingCallback import LossLoggingCallback
from callbacks.ReconstructionCallback import ReconstructionImagesCallback
from models.model_abstract import VAEWrapper
from utils.vae_utils import sampling


class VLAEGAN(VAEWrapper):

    def __init__(self, input_dim: Tuple[int, int, int],
                 inf0_kernels_strides_featuremaps: List[Tuple[int, int, int, int]],
                 inf1_kernels_strides_featuremaps: List[Tuple[int, int, int, int]],
                 ladder0_kernels_strides_featuremaps: List[Tuple[int, int, int, int]],
                 ladder1_kernels_strides_featuremaps: List[Tuple[int, int, int, int]],
                 ladder2_kernels_strides_featuremaps: List[Tuple[int, int, int, int]],
                 gen2_num_units: List[int],
                 gen1_num_units: List[int],
                 gen0_kernels_strides_featuremaps: List[Tuple[int, int, int, int]],
                 use_dropout: bool,
                 use_batch_norm: bool,
                 log_dir: str, kernel_visualization_layer: int, num_samples: int,
                 feature_map_layers: Sequence[int], inner_activation: str, decay_rate: float,
                 feature_map_reduction_factor: int, z_dims: List[int], dropout_rate: float = 0.3):
        """
        Variational autoencoder with adversarial loss, implementation based on
        https://arxiv.org/pdf/1512.09300.pdf and
        https://github.com/baudm/vaegan-celebs-keras/tree/09a012bfecd8d0b202b5531bc646100428d3fa83

        :param input_dim: dimensionality of the input image
        :param inf0_kernels_strides_featuremaps: kernel strides of the first inference network (convolutional part), using a list containing three items adds three convolutional blocks if the other parameters are set accordingly (same list length)
        :param inf1_kernels_strides_featuremaps: kernel strides of the second inference network (convolutional part), using a list containing three items adds three convolutional blocks if the other parameters are set accordingly (same list length)
        :param ladder0_kernels_strides_featuremaps: kernel strides of the first ladder network (convolutional part), using a list containing three items adds three convolutional blocks if the other parameters are set accordingly (same list length)
        :param ladder1_kernels_strides_featuremaps: kernel strides of the second ladder network (convolutional part), using a list containing three items adds three convolutional blocks if the other parameters are set accordingly (same list length)
        :param ladder2_kernels_strides_featuremaps: kernel strides of the third ladder network (convolutionl part), using a list containing three items adds three convolutional blocks if the other parameters are set accordingly (same list length)
        :param gen2_num_units: number of units in the first generator network (fully connected part), using a list containing three items adds three dense layers if the other parameters are set accordingly (same list length)
        :param gen1_num_units: number of units in the second generator network (fully connected part), using a list containing three items adds three dense layers if the other parameters are set accordingly (same list length)
        :param gen0_kernels_strides_featuremaps:  kernel strides of the third generator network (convolutionl part), using a list containing three items adds three convolutional blocks if the other parameters are set accordingly (same list length)
        :param use_dropout:
        :param use_batch_norm:
        :param log_dir:
        :param kernel_visualization_layer: layer index of the layer to visualize the kernels
        :param num_samples: number of generations/reconstructions during training
        :param feature_map_layers: layer indices of layers to visualize feature maps during training
        :param inner_activation: either "ReLU" or "LeakyReLU"
        :param decay_rate: learning rate decay rate
        :param feature_map_reduction_factor: factor by which to reduce feature maps or number of units
        :param z_dims: dimensionality of latent spaces (list of three ints in this case)
        :param dropout_rate:
        """
        super().__init__(input_dim, log_dir, kernel_visualization_layer, num_samples, feature_map_layers,
                         inner_activation, decay_rate, feature_map_reduction_factor, z_dims, ["mu_1", "mu_2", "mu_3"],
                         ["log_var_1", "log_var_2", "log_var_3"])
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.inf0_kernels_strides_featuremaps = inf0_kernels_strides_featuremaps
        self.inf1_kernels_strides_featuremaps = inf1_kernels_strides_featuremaps
        self.ladder0_kernels_strides_featuremaps = ladder0_kernels_strides_featuremaps
        self.ladder1_kernels_strides_featuremaps = ladder1_kernels_strides_featuremaps
        self.ladder2_kernels_strides_featuremaps = ladder2_kernels_strides_featuremaps
        self.gen2_num_units = gen2_num_units
        self.gen1_num_units = gen1_num_units
        self.gen0_kernels_strides_featuremaps = gen0_kernels_strides_featuremaps
        self.dropout_rate = dropout_rate

        def _discriminator(input_shape: Tuple[int, int, int]):
            x = inpt = Input(shape=input_shape)
            x = Conv2D(batch_input_shape=input_shape, filters=128, kernel_size=3)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Conv2D(filters=256, kernel_size=3)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.input_dim[0] > 100:
                x = MaxPool2D()(x)
            x = x_feat = Conv2D(filters=256 if self.input_dim[0] <= 100 else 128, kernel_size=3)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            if self.input_dim[0] > 100:
                x = MaxPool2D()(x)
                x_feat = x
            x = Flatten()(x)
            x = Dense(512)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Dense(1, activation='sigmoid')(x)

            return Model(inpt, [x_feat, x])

        self.discriminator = _discriminator(self.input_dim)

        self._build()

    def _build(self):
        self.inputs = i0 = l0 = Input(self.input_dim)

        # INFERENCE 0
        for kernelsize, poolsize, stride, feature_maps in self.inf0_kernels_strides_featuremaps:
            i0 = Conv2D(filters=math.ceil(feature_maps / self.feature_map_reduction_factor), kernel_size=kernelsize,
                        strides=stride, padding='same')(i0)
            i0 = BatchNormalization()(i0) if self.use_batch_norm else i0
            i0 = ReLU()(i0)
            i0 = MaxPool2D(pool_size=poolsize)(i0)

        # LADDER 0
        for kernelsize, poolsize, stride, feature_maps in self.ladder0_kernels_strides_featuremaps:
            l0 = Conv2D(filters=math.ceil(feature_maps / self.feature_map_reduction_factor), kernel_size=kernelsize,
                        strides=stride, padding='same')(l0)
            l0 = Dropout(self.dropout_rate)(l0) if self.use_dropout else l0
            l0 = BatchNormalization()(l0) if self.use_batch_norm else i0
            l0 = ReLU()(l0)
            l0 = MaxPool2D(pool_size=poolsize)(l0)
        l0 = Flatten()(l0)
        l0 = Dropout(self.dropout_rate)(l0) if self.use_dropout else l0
        self.mu_0 = Dense(self.z_dims[0], name='mu_1')(l0)
        self.log_var_0 = Dense(self.z_dims[0], name='log_var_1')(l0)
        z_0 = Lambda(sampling, name="z_1_latent")([self.mu_0, self.log_var_0])

        # INFERENCE 1
        i1 = i0
        for kernelsize, poolsize, stride, feature_maps in self.inf1_kernels_strides_featuremaps:
            i1 = Conv2D(filters=math.ceil(feature_maps / self.feature_map_reduction_factor), kernel_size=kernelsize,
                        strides=stride, padding='same')(i1)
            i1 = Dropout(self.dropout_rate)(i1) if self.use_dropout else i1
            i1 = BatchNormalization()(i1) if self.use_batch_norm else i1
            i1 = ReLU()(i1)
            i1 = MaxPool2D(pool_size=poolsize)(i1)

        # LADDER 1
        l1 = i0
        for kernelsize, poolsize, stride, feature_maps in self.ladder1_kernels_strides_featuremaps:
            l1 = Conv2D(filters=math.ceil(feature_maps / self.feature_map_reduction_factor), kernel_size=kernelsize,
                        strides=stride, padding='same')(l1)
            l1 = Dropout(self.dropout_rate)(l1) if self.use_dropout else l1
            l1 = BatchNormalization()(l1) if self.use_batch_norm else l1
            l1 = ReLU()(l1)
            l1 = MaxPool2D(pool_size=poolsize)(l1)
        l1 = Flatten()(l1)
        l1 = Dropout(self.dropout_rate)(l1) if self.use_dropout else l1
        self.mu_1 = Dense(self.z_dims[1], name='mu_2')(l1)
        self.log_var_1 = Dense(self.z_dims[1], name='log_var_2')(l1)
        z_1 = Lambda(sampling, name="z_2_latent")([self.mu_1, self.log_var_1])

        # LADDER 2
        l2 = i1
        for kernelsize, poolsize, stride, feature_maps in self.ladder2_kernels_strides_featuremaps:
            l2 = Conv2D(filters=math.ceil(feature_maps / self.feature_map_reduction_factor), kernel_size=kernelsize,
                        strides=stride, padding='same')(l2)
            l2 = Dropout(self.dropout_rate)(l2) if self.use_dropout else l2
            l2 = BatchNormalization()(l2)
            l2 = ReLU()(l2)
            l2 = MaxPool2D(pool_size=poolsize)(l2)
        shape_before_flattening = K.int_shape(l2)[1:]
        l2 = Flatten()(l2)
        l2 = Dropout(self.dropout_rate)(l2) if self.use_dropout else l2
        self.mu_2 = Dense(self.z_dims[2], name='mu_3')(l2)
        self.log_var_2 = Dense(self.z_dims[2], name='log_var_3')(l2)
        z_2 = Lambda(sampling, name="z_3_latent")([self.mu_2, self.log_var_2])

        encoder_output = [z_0, z_1, z_2]

        self.encoder = Model(self.inputs, encoder_output, name='encoder')

        ### DECODER ###

        z_1_input, z_2_input, z_3_input = Input((self.z_dims[0],), name='z_1'), Input((self.z_dims[1],),
                                                                                      name='z_2'), Input(
            (self.z_dims[2],), name='z_3')

        # GENERATIVE 2
        g2 = z_3_input
        for num_units in self.gen2_num_units:
            g2 = Dense(math.ceil(num_units / self.feature_map_reduction_factor))(g2)
            g2 = Dropout(self.dropout_rate)(g2) if self.use_dropout else g2
            g2 = BatchNormalization()(g2) if self.use_batch_norm else g2
            g2 = ReLU()(g2)

        # GENERATIVE 1
        g1 = Concatenate()([g2, z_2_input])
        for num_units in self.gen1_num_units:
            g1 = Dense(math.ceil(num_units / self.feature_map_reduction_factor))(g1)
            g1 = Dropout(self.dropout_rate)(g1) if self.use_dropout else g1
            g1 = BatchNormalization()(g1) if self.use_batch_norm else g1
            g1 = ReLU()(g1)

        # GENERATIVE 0
        g0 = Concatenate()([g1, z_1_input])
        g0 = Dense(np.prod(shape_before_flattening))(g0)
        g0 = Dropout(self.dropout_rate)(g0) if self.use_dropout else g0
        g0 = BatchNormalization()(g0) if self.use_batch_norm else g0
        g0 = ReLU()(g0)
        g0 = Reshape(shape_before_flattening)(g0)
        for i, (kernelsize, poolsize, stride, feature_maps) in enumerate(self.gen0_kernels_strides_featuremaps):
            g0 = UpSampling2D(size=poolsize)(g0)
            g0 = Conv2DTranspose(filters=math.ceil(feature_maps / self.feature_map_reduction_factor),
                                 kernel_size=kernelsize, strides=stride, padding='same')(g0)
            g0 = Dropout(self.dropout_rate)(g0) if self.use_dropout else g0
            g0 = BatchNormalization()(g0) if i < len(
                self.gen0_kernels_strides_featuremaps) - 1 and self.use_batch_norm else g0
            g0 = ReLU()(g0) if i < len(self.gen0_kernels_strides_featuremaps) - 1 else g0
        g0 = Activation('sigmoid')(g0)

        self.decoder = Model([z_1_input, z_2_input, z_3_input], g0, name='decoder')
        decoder_output = self.decoder(encoder_output)

        self.model = Model(self.inputs, decoder_output, name='vlae')

        #####

        self.x_tilde = self.decoder(encoder_output)
        self.x_p = self.decoder(self.decoder.inputs)

        self.dis_x_feat, self.dis_x = self.discriminator(self.inputs)
        self.dis_x_feat_tilde, self.dis_x_tilde = self.discriminator(self.x_tilde)
        self.dis_x_feat_p, self.dis_x_p = self.discriminator(self.x_p)

        return self.encoder

    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        def mean_gaussian_negative_log_likelihood(y_true, y_pred):
            nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
            axis = tuple(range(1, len(K.int_shape(y_true))))
            return K.mean(K.sum(nll, axis=axis), axis=-1)

        def vae_kl_loss():
            kl_loss = 0.0
            for lv, m in zip([self.log_var_0, self.log_var_1, self.log_var_2], [self.mu_0, self.mu_1, self.mu_2]):
                kl_loss += -0.5 * K.sum(1 + lv - K.square(m) - K.exp(lv), axis=1)
            return kl_loss

        dis_nl_loss = mean_gaussian_negative_log_likelihood(self.dis_x_feat, self.dis_x_feat_tilde)
        kl_loss = vae_kl_loss()

        self.encoder_train = Model(self.inputs, self.dis_x_feat_tilde)
        self.encoder_train.add_loss(kl_loss)
        self.encoder_train.add_loss(dis_nl_loss)

        self.decoder_train = Model([self.inputs, *self.decoder.inputs], [self.dis_x_tilde, self.dis_x_p])
        self.decoder_train.add_loss(0.75 * dis_nl_loss)

        self.discriminator_train = Model([self.inputs, *self.decoder.inputs],
                                         [self.dis_x, self.dis_x_tilde, self.dis_x_p])

        self.vlae_gan = Model(self.inputs, self.dis_x_tilde)

    def train(self, x_train: Union[Iterator, np.ndarray], batch_size, epochs, weights_folder, print_every_n_batches=100,
              initial_epoch=0, lr_decay=1, embedding_samples: int = 5000, y_train: Optional[np.ndarray] = None,
              x_test: Optional[Union[Iterator, np.ndarray]] = None, y_test: Optional[np.ndarray] = None,
              steps_per_epoch: int = None, **kwargs):

        embedding_callback_params = kwargs.get('embedding_callback_params', {})

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

        # checkpoint2 = ModelCheckpoint(os.path.join(weights_folder, 'weights.h5'), save_weights_only=True, verbose=1)
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
        av_encoder_callback = ActivationVisualizationCallback(log_dir=self.log_dir, model=self.encoder,
                                                              model_name='encoder',
                                                              x_train=x_train_subset,
                                                              print_every_n_batches=print_every_n_batches)
        av_decoder_callback = ActivationVisualizationCallback(log_dir=self.log_dir, model=self.decoder,
                                                              model_name='decoder',
                                                              x_train=x_train_subset,
                                                              print_every_n_batches=print_every_n_batches,
                                                              x_train_transform=lambda x, m: m.predict(x),
                                                              transform_params=[self.encoder])
        hs_callback = HiddenSpaceCallback(log_dir=self.log_dir, vae=self, batch_size=batch_size,
                                          x_train=x_train_subset, y_train=y_embedding, max_samples=5000,
                                          layer_names=self.mu_layer_names, plot_params=embedding_callback_params)
        ll_callback = LossLoggingCallback(logdir=self.log_dir)
        # tb_callback has to be first as we use its filewriter subsequently but it is initialized by keras in this given order
        callbacks_list = [av_encoder_callback, av_decoder_callback, hs_callback, fm_callback, rc_callback, ll_callback]

        logging.info("Training for {} epochs".format(epochs))

        optimizer = Adam(lr=0.00005, beta_1=0.5, decay=self.decay_rate)

        def set_trainable(model: Model, trainable: bool):
            """
            Set layers of model to trainable / not trainable as specified by 'trainable' (in place)
            :param model: the model to manipulate
            :param trainable:
            :return: nothing
            """
            model.trainable = trainable
            for layer in model.layers:
                layer.trainable = trainable

        set_trainable(self.encoder, False)
        set_trainable(self.decoder, False)
        set_trainable(self.discriminator, True)
        self.discriminator_train.compile(optimizer, ['binary_crossentropy'] * 3, ['acc'] * 3)
        self.discriminator_train.summary()

        set_trainable(self.discriminator, False)
        set_trainable(self.decoder, True)
        self.decoder_train.compile(optimizer, ['binary_crossentropy'] * 2, ['acc'] * 2)
        self.decoder_train.summary()

        set_trainable(self.decoder, False)
        set_trainable(self.encoder, True)
        self.encoder_train.compile(optimizer)
        self.encoder_train.summary()

        set_trainable(self.vlae_gan, True)

        models = [self.discriminator_train, self.decoder_train, self.encoder_train]
        epoch = initial_epoch
        for cb in callbacks_list:
            cb.set_model(self.model)
            cb.on_train_begin()
        if x_test is not None:
            x_test = np.copy(x_test)
            logging.info("Evaluating models")
            if isinstance(x_test, Iterator):
                raise RuntimeError("Model evaluation currently is only supported for numpy test data.")
            assert isinstance(x_test, np.ndarray)
            # make sure len(x_test) is multiple of batch_size
            test_batch_size = 32
            test_size = x_test.shape[0]
            num_batches = test_size // test_batch_size
            x_test = x_test[:num_batches * test_batch_size]
            test_size = x_test.shape[0]
        while epoch < epochs:
            batch_index = 0
            for cb in callbacks_list:
                cb.on_epoch_begin(epoch)

            if isinstance(x_train, Iterator):
                steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else x_train.n // batch_size
            else:
                steps_per_epoch = len(x_train) // batch_size
            with tqdm(total=steps_per_epoch,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | {postfix[0]} {postfix[1][value]} | {postfix[2]} {postfix[3][value]:.3f} | {postfix[4]} {postfix[5][value]:.3f}",
                      postfix=["Discriminator", dict(value=0), "Decoder", dict(value=0), "Encoder",
                               dict(value=0)]) as t:
                for steps_done in range(steps_per_epoch):
                    for cb in callbacks_list:
                        cb.on_batch_begin(batch_index, {})
                    losses = {}
                    if isinstance(x_train, Iterator):
                        x, y = next(x_train)
                    else:
                        x, y = (x_train[batch_index * batch_size:batch_index * batch_size + batch_size],
                                y_train[batch_index * batch_size:batch_index * batch_size + batch_size])
                    if x.shape[0] != batch_size:
                        steps_done += 1
                        continue

                    for model in models:
                        if model == self.discriminator_train:
                            xi = [x, *[np.random.normal(size=(batch_size, z)) for z in self.z_dims]]
                            y_real = np.ones((batch_size,), dtype='float32')
                            y_fake = np.zeros((batch_size,), dtype='float32')
                            yi = [y_real, y_fake, y_fake]
                            then = time.time()
                            outs = model.train_on_batch(xi, yi)
                            now = time.time()

                            print("Training discriminator took {} seconds".format(int(now - then)))
                            t.postfix[1]["value"] = "{:.3f} {:.3f} {:.3f}".format(outs[4],
                                                                                  outs[7 if len(outs) > 7 else 5],
                                                                                  outs[10 if len(outs) > 7 else 6])
                            losses['discriminator_loss'] = outs[0]
                            losses['discriminator_loss_x_true'] = outs[1]
                            losses['discriminator_loss_x_reconstr'] = outs[2]
                            losses['discriminator_loss_x_sampling'] = outs[3]
                            losses['discriminator_acc_x_true'] = outs[4]
                            losses['discriminator_acc_x_reconstr'] = outs[7 if len(outs) > 7 else 5]
                            losses['discriminator_acc_x_sampling'] = outs[10 if len(outs) > 7 else 6]
                        elif model == self.decoder_train:
                            xi = [x, *[np.random.normal(size=(batch_size, z)) for z in self.z_dims]]
                            y_real = np.ones((batch_size,), dtype='float32')
                            yi = [y_real, y_real]
                            then = time.time()
                            outs = model.train_on_batch(xi, yi)
                            now = time.time()

                            print("Training decoder took {} seconds".format(int(now - then)))
                            t.postfix[3]["value"] = outs[3]
                            losses['decoder_acc'] = outs[3]
                            losses['decoder_loss'] = outs[0]
                        elif model == self.encoder_train:
                            then = time.time()
                            outs = model.train_on_batch(x, None)
                            now = time.time()
                            print("Training encoder took {} seconds".format(int(now - then)))
                            mean_loss = np.mean(outs)
                            t.postfix[5]["value"] = mean_loss
                            losses['encoder_loss'] = mean_loss
                    t.update()
                    for cb in callbacks_list:
                        cb.on_batch_end(batch_index, losses)
                    batch_index += 1
            losses = {}
            if x_test is not None:
                y_real = np.ones((test_size,), dtype='float32')
                y_fake = np.zeros((test_size,), dtype='float32')
                for model in models:
                    if model == self.discriminator_train:
                        x = [x_test, *[np.random.normal(size=(test_size, z)) for z in self.z_dims]]
                        y = [y_real, y_fake, y_fake]
                        outs = model.evaluate(x, y, batch_size=test_batch_size)
                        losses['discriminator_loss'] = outs[0]
                        losses['discriminator_loss_x_true'] = outs[1]
                        losses['discriminator_loss_x_reconstr'] = outs[2]
                        losses['discriminator_loss_x_sampling'] = outs[3]
                        losses['discriminator_acc_x_true'] = outs[4]
                        losses['discriminator_acc_x_reconstr'] = outs[7 if len(outs) > 7 else 5]
                        losses['discriminator_acc_x_sampling'] = outs[10 if len(outs) > 7 else 6]
                    elif model == self.decoder_train:
                        x = [x_test, *[np.random.normal(size=(test_size, z)) for z in self.z_dims]]
                        y = [y_real, y_real]
                        outs = model.evaluate(x, y, batch_size=test_batch_size)
                        losses['decoder_acc'] = outs[3]
                        losses['decoder_loss'] = outs[0]
                    elif model == self.encoder_train:
                        outs = model.evaluate(x_test, None, batch_size=test_batch_size)
                        mean_loss = np.mean(outs)
                        losses['encoder_loss'] = mean_loss
            for cb in callbacks_list:
                cb.on_epoch_end(epoch, losses)
                self.model.save(os.path.join(self.log_dir, 'weights.h5'))
            epoch += 1

        for cb in callbacks_list:
            cb.on_train_end()
