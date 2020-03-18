import argparse
import csv
import datetime
import logging
import os
import sys
import traceback
from argparse import ArgumentParser
from random import random, randint
from shutil import rmtree
from typing import List, Tuple

import numpy as np
import sklearn
from keras.datasets import cifar10, mnist
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

from models.AlexAlexNetVAE import AlexAlexNetVAE
from models.AlexNet import AlexNet
from models.AlexNetVAE import AlexNetVAE
from models.FrozenAlexNetVAE import FrozenAlexNetVAE
from models.HVAE import HVAE
from models.SimpleClassifier import SimpleClassifier
from models.VLAE import VLAE
from models.model_abstract import DeepCNNClassifierWrapper, VAEWrapper
from utils.img_ops import resize_array


def main(args: List[str]):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format="'[%(asctime)s][%(filename)s][%(levelname)s]: %(message)s'",
                        filename='{}.log'.format(datetime.date.today()),
                        level=logging.INFO, filemode="a+")

    parser = ArgumentParser(description='Functionality for Leonards master thesis')
    parser.add_argument('--configuration', type=str,
                        choices=['vanilla_vae', 'large_vanilla_vae', 'alexnet_classifier',
                                 'simple_classifier', 'alexnet_vae', 'frozen_alexnet_vae',
                                 'alexnet_vae_classification_loss', 'vlae', 'hvae'],
                        help="The configuration to execute.\n\n"
                             "mnist: VAE trained on mnist\n"
                             "cifar10_vae: VAE trained on cifar10\n"
                             "celeba: VAE trained on (128,128) sized celeba dataset\n"
                             "celeba_large_model: celeba trained on upscaled (224,224) sized images\n"
                             "imagenet_classification_alexnet: train AlexNet on imagenet classification task\n"
                             "alexnet_vae_classification_loss: train normal AlexNet but reconstruction loss by difference in AlexNet classification",
                        required=True)
    parser.add_argument('--data_path',
                        help="The path containing the individual datafolders. If the path to the imagenet folder is "
                             "/foo/bar/imagenet, the value should be /foo/bar/. Can be absolute or relative. Required for CelebA, Imagenet, and MNIST.",
                        required=False)
    parser.add_argument('--feature_map_layers', nargs='+', type=int,
                        help="The indices of layers after which to compute the "
                             "feature maps. Exemplary input: 1 2 4 8 13", required=False)
    parser.add_argument('--kernel_visualization_layer', type=int,
                        help="The index of layer after which to compute the max stimuli.", default=-1)
    parser.add_argument('--logdir', type=str, default="logs/",
                        help="The directory to which the logs will be written. Absolute or relative.")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size.")
    parser.add_argument('--num_epochs', type=int, default=100, help="The number of epochs.")
    parser.add_argument('--initial_epoch', type=int, default=0, help="The initial epochs (0 for a new run).")
    parser.add_argument('--print_every_n_batches', type=int, default=100,
                        help="After how many batches the callbacks will be executed.")
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="The initial learning rate passed to the optimizer.")
    parser.add_argument('--r_loss_factor', type=float, default=10E4,
                        help="The factor by which to weigh the reconstruction loss in case of variational autoencoder")
    parser.add_argument('--mode', type=str, choices=['build', 'load'], default='build',
                        help="Whether to build a new model or load an existing one.")
    parser.add_argument('--z_dims', nargs='+', type=int,
                        help="The dimensionalities of the embedding spaces", default=[10], required=False)
    parser.add_argument('--use_batch_norm', type=str2bool, default=False)
    parser.add_argument('--use_dropout', type=str2bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--steps_per_epoch', type=int, required=False)
    parser.add_argument('--weights_path', type=str, required=False,
                        help="The path to restore the model from. Is only considered if --mode=load You usually might "
                             "want to also set initial_epoch.")
    parser.add_argument('--alexnet_weights_path', type=str,
                        help="Only for configuration 'frozen_alexnet_vae'. The path to (and including) the .h5 file "
                             "to restore the AlexNet classifier from.")
    parser.add_argument('--dataset', type=str, choices=['celeba', 'imagenet', 'cifar10', 'mnist'],
                        help="Which dataset to use for training. WARNING: Use ImageNet for classification.")
    parser.add_argument('--use_fc', type=str2bool, default=True, help="Whether to use the fully connected layers in "
                                                                      "AlexNet or not.")
    parser.add_argument('--inner_activation', type=str, choices=["ReLU", "LeakyReLU"], default="ReLU",
                        help="The activation functions used inside the model (output activation usually won't be "
                             "affected by this setting).")
    parser.add_argument('--lr_decay', type=float, default=1e-7,
                        help="The learning rate decay. Should be in interval [0,1].")
    parser.add_argument('--rgb', type=str2bool, default=True)
    parser.add_argument('--feature_map_reduction_factor', type=int, default=1,
                        help="The factor by which to reduce the number of feature maps. If a layer usually has 50 "
                             "feature maps, setting a factor to 2 will yield only 25 feature maps.")
    args = parser.parse_args(args)

    dataset_subfolder = 'celeb' if args.dataset == 'celeba' else 'imagenet/ILSVRC/Data/CLS-LOC/'

    logging.info("Program called with arguments: {}".format(sys.argv))

    weights = os.path.join(args.logdir, 'weights')

    if not os.path.exists(args.logdir):
        logging.info("Creating logdir: {}".format(args.logdir))
        os.mkdir(args.logdir)
    elif not len(os.listdir(args.logdir)) == 0:
        logging.warning("Logdir not empty. Deleting content...")
        rmtree(args.logdir)
        os.mkdir(args.logdir)

    if not os.path.exists(weights):
        logging.info("Creating weights dir: {}".format(weights))
        os.mkdir(weights)

    input_dim = None
    x_train, y_train = None, None
    x_val, y_val = None, None
    embedding_callback_params = []

    if args.configuration == 'simple_classifier':
        input_dim = infer_input_dim((32, 32), args)
        model = SimpleClassifier(input_dim=input_dim, encoder_conv_filters=[32, 64, 64, 64],
                                 encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[1, 2, 2, 1],
                                 log_dir=args.logdir, use_batch_norm=args.use_batch_norm, use_dropout=args.use_dropout,
                                 inner_activation=args.inner_activation, feature_map_layers=args.feature_map_layers,
                                 feature_map_reduction_factor=args.feature_map_reduction_factor, num_classes=10,
                                 decay_rate=args.lr_decay, kernel_visualization_layer=args.kernel_visualization_layer,
                                 num_samples=args.num_samples, dropout_rate=args.dropout_rate)

    elif args.configuration == 'vanilla_vae':
        from models.VAE import VariationalAutoencoder
        input_dim = infer_input_dim((128, 128), args)
        model = VariationalAutoencoder(input_dim=input_dim, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[2, 2, 2, 2],
                                       decoder_conv_t_filters=[64, 64, 32, 3 if args.rgb else 1],
                                       decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                       decoder_conv_t_strides=[2, 2, 2, 2], log_dir=args.logdir, z_dims=args.z_dims,
                                       kernel_visualization_layer=args.kernel_visualization_layer,
                                       feature_map_layers=args.feature_map_layers, use_batch_norm=args.use_batch_norm,
                                       decay_rate=args.lr_decay, num_samples=args.num_samples,
                                       feature_map_reduction_factor=args.feature_map_reduction_factor,
                                       inner_activation=args.inner_activation, dropout_rate=args.dropout_rate)
    elif args.configuration == 'large_vanilla_vae':
        from models.VAE import VariationalAutoencoder
        input_dim = infer_input_dim((224, 224), args)
        model = VariationalAutoencoder(input_dim=input_dim, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[11, 7, 5, 3], encoder_conv_strides=[4, 2, 2, 2],
                                       decoder_conv_t_filters=[64, 64, 32, 3 if args.rgb else 1],
                                       decoder_conv_t_kernel_size=[3, 5, 7, 11],
                                       decoder_conv_t_strides=[2, 2, 2, 4], log_dir=args.logdir, z_dims=args.z_dims,
                                       use_batch_norm=args.use_batch_norm, use_dropout=args.use_dropout,
                                       kernel_visualization_layer=args.kernel_visualization_layer,
                                       num_samples=args.num_samples, dropout_rate=args.dropout_rate,
                                       inner_activation=args.inner_activation, decay_rate=args.lr_decay,
                                       feature_map_layers=args.feature_map_layers,
                                       feature_map_reduction_factor=args.feature_map_reduction_factor)
    elif args.configuration == 'alexnet_classifier':
        input_dim = infer_input_dim((224, 224), args)
        model = AlexNet(input_dim=input_dim, log_dir=args.logdir, feature_map_layers=args.feature_map_layers,
                        use_batch_norm=args.use_batch_norm, decay_rate=args.lr_decay,
                        inner_activation=args.inner_activation,
                        kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                        use_fc=args.use_fc, feature_map_reduction_factor=args.feature_map_reduction_factor)
    elif args.configuration == 'alexnet_vae':
        input_dim = infer_input_dim((224, 224), args)
        model = AlexNetVAE(input_dim=input_dim, log_dir=args.logdir, z_dims=args.z_dims,
                           feature_map_layers=args.feature_map_layers, use_batch_norm=args.use_batch_norm,
                           kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                           use_fc=args.use_fc, inner_activation=args.inner_activation, decay_rate=args.lr_decay,
                           feature_map_reduction_factor=args.feature_map_reduction_factor)
    elif args.configuration == 'alexnet_vae_classification_loss':
        input_dim = infer_input_dim((224, 224), args)
        model = AlexAlexNetVAE(input_dim=input_dim, log_dir=args.logdir, z_dims=args.z_dims,
                               feature_map_layers=args.feature_map_layers, use_batch_norm=args.use_batch_norm,
                               kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                               use_fc=args.use_fc, inner_activation=args.inner_activation, decay_rate=args.lr_decay,
                               feature_map_reduction_factor=args.feature_map_reduction_factor,
                               alexnet_weights_path=args.alexnet_weights_path)
    elif args.configuration == 'frozen_alexnet_vae':
        input_dim = infer_input_dim((224, 224), args)

        # TODO remove static requirements
        shape_before_flattening = (7, 7, 256)
        model = FrozenAlexNetVAE(z_dims=args.z_dims, use_dropout=args.use_dropout, dropout_rate=args.dropout_rate,
                                 use_batch_norm=args.use_batch_norm, shape_before_flattening=shape_before_flattening,
                                 input_dim=input_dim, log_dir=args.logdir, weights_path=args.alexnet_weights_path,
                                 kernel_visualization_layer=args.kernel_visualization_layer, decay_rate=args.lr_decay,
                                 inner_activation=args.inner_activation,
                                 feature_map_reduction_factor=args.feature_map_reduction_factor,
                                 feature_map_layers=args.feature_map_layers, num_samples=args.num_samples)

    elif args.configuration == 'vlae':
        input_dim = infer_input_dim((28, 28), args)
        model = VLAE(input_dim=input_dim, log_dir=args.logdir,
                     kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                     feature_map_layers=args.feature_map_layers, inner_activation=args.inner_activation,
                     decay_rate=args.lr_decay, feature_map_reduction_factor=args.feature_map_reduction_factor,
                     z_dims=args.z_dims, dropout_rate=args.dropout_rate)
    elif args.configuration == 'hvae':
        input_dim = infer_input_dim((28, 28), args)
        model = HVAE(input_dim=input_dim, log_dir=args.logdir,
                     kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                     feature_map_layers=args.feature_map_layers, inner_activation=args.inner_activation,
                     decay_rate=args.lr_decay, feature_map_reduction_factor=args.feature_map_reduction_factor,
                     z_dims=args.z_dims, dropout_rate=args.dropout_rate)
    if args.dataset == 'cifar10':
        (x_train, y_train), (x_val, y_val) = cifar10.load_data()

        x_train = resize_array(x_train, input_dim[:2], args.rgb)
        x_val = resize_array(x_val, input_dim[:2], args.rgb)

        x_train = x_train.astype('float32') / 255.
        x_val = x_val.astype('float32') / 255.

        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)
        if len(x_val.shape) == 3:
            x_val = np.expand_dims(x_val, -1)

        if isinstance(model, DeepCNNClassifierWrapper):
            y_train = y_train.squeeze()
            y_val = y_val.squeeze()
            y_train = to_categorical(y_train, num_classes=10)
            y_val = to_categorical(y_val, num_classes=10)
    elif args.dataset == 'mnist':
        seed = randint(0, 2 ** 32 - 1)
        (x_train, y_train), (x_val, y_val) = mnist.load_data()

        x_train, y_train = shuffle(x_train, y_train, random_state=seed)
        x_val, y_val = shuffle(x_val, y_val, random_state=seed)

        x_train = resize_array(x_train, input_dim[:2], args.rgb)
        x_val = resize_array(x_val, input_dim[:2], args.rgb)

        x_train = x_train.astype('float32') / 255.
        x_val = x_val.astype('float32') / 255.

        # read morpho mnist
        morpho_headers = []
        morpho_train = {}
        morpho_test = {}
        with open(os.path.join(args.data_path, 'train-morpho.csv'), newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar="'")
            for i, row in enumerate(spamreader):
                if i == 0:
                    for header in row:
                        morpho_train.setdefault(header, [])
                        morpho_headers.append(header)
                else:
                    for j, cell in enumerate(row):
                        morpho_train[morpho_headers[j]].append(float(cell))

        with open(os.path.join(args.data_path, 't10k-morpho.csv'), newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar="'")
            for i, row in enumerate(spamreader):
                if i == 0:
                    for header in row:
                        morpho_test.setdefault(header, [])
                else:
                    for j, cell in enumerate(row):
                        morpho_test[morpho_headers[j]].append(float(cell))

        # normalize all values to [0,1]
        for morpho_dict in [morpho_train, morpho_test]:
            for k, v in morpho_dict.items():
                v = np.array(v)
                v = (v - v.min()) / (v.max() - v.min())
                morpho_dict[k] = shuffle(v, random_state=seed)

        x_train = x_train
        y_train = y_train
        x_val = x_val
        y_val = y_val

        bins = np.arange(0.0, 1.0, 0.1)
        embedding_callback_params += [
            {
                'c': np.digitize(morpho_train['slant'], bins),
                'label': 'Slant',
            },
            {
                'c': np.digitize(morpho_train['thickness'], bins),
                'label': 'Thickness',
            },
            {
                'c': np.digitize(morpho_train['area'], bins),
                'label': 'Area'
            },
            {
                'c': np.digitize(morpho_train['length'], bins),
                'label': 'Length'
            },
            {
                'c': np.digitize(morpho_train['width'], bins),
                'label': 'Width'
            },
            {
                'c': np.digitize(morpho_train['height'], bins),
                'label': 'Height'
            },
            {
                'c': y_train,
                'label': 'Class Identity'
            }
        ]

        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)
        if len(x_val.shape) == 3:
            x_val = np.expand_dims(x_val, -1)

        if isinstance(model, DeepCNNClassifierWrapper):
            y_train = y_train.squeeze()
            y_val = y_val.squeeze()
            y_train = to_categorical(y_train, num_classes=10)
            y_val = to_categorical(y_val, num_classes=10)
    elif args.dataset == 'celeba':
        # TODO variable validation split size
        data_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)
        x_train = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder),
            target_size=input_dim[:2], batch_size=args.batch_size,
            class_mode='input', interpolation='lanczos',
            follow_links=True, subset='training', color_mode='rgb' if args.rgb else 'grayscale')
        y_train = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder),
            target_size=input_dim[:2], batch_size=args.batch_size,
            class_mode='input', interpolation='lanczos',
            follow_links=True, subset='validation', color_mode='rgb' if args.rgb else 'grayscale')
    elif args.dataset == 'imagenet':
        if isinstance(model, DeepCNNClassifierWrapper):
            class_mode = 'categorical'
        elif isinstance(model, VAEWrapper):
            class_mode = 'input'
        else:
            raise AttributeError("Unrecognized model class '{}'".format(model.__class__))
        # TODO variable validation split size
        data_gen = ImageDataGenerator(rescale=1. / 255)
        x_train = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder, 'train'),
            target_size=input_dim[:2], batch_size=args.batch_size,
            class_mode=class_mode, interpolation='lanczos',
            follow_links=True, subset='training', color_mode='rgb' if args.rgb else 'grayscale')
        y_train = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder, 'val'),
            target_size=input_dim[:2], batch_size=args.batch_size,
            class_mode=class_mode, interpolation='lanczos',
            follow_links=True, subset='validation', color_mode='rgb' if args.rgb else 'grayscale')

    if args.mode == 'build':
        model.save()
    else:
        model.load_weights(args.weights_path)

    model.compile(args.learning_rate, args.r_loss_factor)

    model.model.summary()
    try:
        model.encoder.summary()
        model.encoder.summary(print_fn=lambda x: logging.info(x))
        model.decoder.summary()
        model.decoder.summary(print_fn=lambda x: logging.info(x))
    except:
        pass
    model.model.summary(print_fn=lambda x: logging.info(x))

    try:
        model.train(x_train=x_train, y_train=y_train, epochs=args.num_epochs,
                    weights_folder=weights,
                    batch_size=args.batch_size,
                    print_every_n_batches=args.print_every_n_batches, initial_epoch=args.initial_epoch,
                    x_test=x_val, y_test=y_val, steps_per_epoch=args.steps_per_epoch,
                    embedding_callback_params=embedding_callback_params)
    except Exception as e:
        logging.error("An error occurred during training:")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
            logging.error(line)
        raise e


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def infer_input_dim(size: Tuple[int, int], args: argparse.Namespace) -> Tuple[int, int, int]:
    if args.rgb:
        return (*size, 3)
    else:
        return (*size, 1)


if __name__ == '__main__':
    main(sys.argv[1:])
