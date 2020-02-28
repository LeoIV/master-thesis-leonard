import argparse
import datetime
import logging
import os
import sys
import traceback
from argparse import ArgumentParser
from shutil import rmtree

import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from models.AlexAlexNetVAE import AlexAlexNetVAE
from models.AlexNet import AlexNet
from models.AlexNetVAE import AlexNetVAE
from models.FrozenAlexNetVAE import FrozenAlexNetVAE
from models.SimpleClassifier import SimpleClassifier
from utils.loaders import load_mnist


def main():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format="'[%(asctime)s][%(filename)s][%(levelname)s]: %(message)s'",
                        filename='{}.log'.format(datetime.date.today()),
                        level=logging.INFO, filemode="a+")

    parser = ArgumentParser(description='Functionality for Leonards master thesis')
    parser.add_argument('--configuration', type=str,
                        choices=['mnist', 'celeba', 'celeba_large_model', 'imagenet_classification_alexnet',
                                 'cifar10_classifier', 'alexnet_vae', 'frozen_alexnet_vae',
                                 'alexnet_vae_classification_loss', 'cifar10_vae'],
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
                             "/foo/bar/imagenet, the value should be /foo/bar/. Can be absolute or relative. Required for CelebA and Imagenet.",
                        required=False)
    parser.add_argument('--feature_map_layers', nargs='+', type=int,
                        help="The indices of layers after which to compute the "
                             "feature maps. Exemplary input: 1 2 4 8 13", required=True)
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
    parser.add_argument('--z_dim', type=int, default=200,
                        help="The embedding space dimensionality. Only considered for VAEs.")
    parser.add_argument('--use_batch_norm', type=str2bool, default=False)
    parser.add_argument('--use_dropout', type=str2bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--weights_path', type=str, required=False,
                        help="The path to restore the model from. Is only considered if --mode=load You usually might "
                             "want to also set initial_epoch.")
    parser.add_argument('--alexnet_weights_path', type=str,
                        help="Only for configuration 'frozen_alexnet_vae'. The path to (and including) the .h5 file "
                             "to restore the AlexNet classifier from.")
    parser.add_argument('--dataset', type=str, choices=['celeba', 'imagenet'],
                        help="Which dataset to use for training. WARNING: Use ImageNet for classification.")
    parser.add_argument('--use_fc', type=str2bool, default=True, help="Whether to use the fully connected layers in "
                                                                      "AlexNet or not.")
    parser.add_argument('--inner_activation', type=str, choices=["ReLU", "LeakyReLU"], default="ReLU",
                        help="The activation functions used inside the model (output activation usually won't be "
                             "affected by this setting).")
    parser.add_argument('--lr_decay', type=float, default=1e-7,
                        help="The learning rate decay. Should be in interval [0,1].")
    parser.add_argument('--feature_map_reduction_factor', type=int, default=1,
                        help="The factor by which to reduce the number of feature maps. If a layer usually has 50 "
                             "feature maps, setting a factor to 2 will yield only 25 feature maps.")
    args = parser.parse_args()

    dataset_subfolder = 'celeb' if args.dataset == 'celeba' else 'imagenet/ILSVRC/Data/CLS-LOC/train'

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

    training_data, training_labels = None, None

    if args.configuration == 'mnist':
        from models.VAE import VariationalAutoencoder
        INPUT_DIM = (28, 28, 1)
        model = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[1, 2, 2, 1],
                                       decoder_conv_t_filters=[64, 64, 32, 1], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                       decoder_conv_t_strides=[1, 2, 2, 1], log_dir=args.logdir, z_dim=args.z_dim,
                                       decay_rate=args.lr_decay, inner_activation=args.inner_activation,
                                       feature_map_layers=args.feature_map_layers, dropout_rate=args.dropout_rate,
                                       num_samples=args.num_samples, use_batch_norm=args.use_batch_norm,
                                       feature_map_reduction_factor=args.feature_map_reduction_factor,
                                       kernel_visualization_layer=args.kernel_visualization_layer)
        (training_data, _), (_, _) = load_mnist()

    elif args.configuration == 'cifar10_vae':
        from models.VAE import VariationalAutoencoder
        INPUT_DIM = (32, 32, 3)
        model = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[1, 2, 2, 1],
                                       decoder_conv_t_filters=[64, 64, 32, 3], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                       decoder_conv_t_strides=[1, 2, 2, 1], log_dir=args.logdir, z_dim=args.z_dim,
                                       decay_rate=args.lr_decay, inner_activation=args.inner_activation,
                                       feature_map_layers=args.feature_map_layers, dropout_rate=args.dropout_rate,
                                       num_samples=args.num_samples, use_batch_norm=args.use_batch_norm,
                                       feature_map_reduction_factor=args.feature_map_reduction_factor,
                                       kernel_visualization_layer=args.kernel_visualization_layer)
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        training_data = np.concatenate([x_train, x_test])
    elif args.configuration == 'cifar10_classifier':
        from models.VAE import VariationalAutoencoder
        INPUT_DIM = (32, 32, 3)
        model = SimpleClassifier(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                 encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[1, 2, 2, 1],
                                 log_dir=args.logdir, use_batch_norm=args.use_batch_norm, use_dropout=args.use_dropout,
                                 inner_activation=args.inner_activation, feature_map_layers=args.feature_map_layers,
                                 feature_map_reduction_factor=args.feature_map_reduction_factor, num_classes=10,
                                 decay_rate=args.lr_decay, kernel_visualization_layer=args.kernel_visualization_layer,
                                 num_samples=args.num_samples, dropout_rate=args.dropout_rate)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        training_data = np.concatenate([x_train, x_test])
        training_labels = np.concatenate([y_train, y_test]).squeeze()

        training_labels = to_categorical(training_labels, num_classes=10)

    elif args.configuration == 'celeba':
        from models.VAE import VariationalAutoencoder
        INPUT_DIM = (128, 128, 3)
        model = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[2, 2, 2, 2],
                                       decoder_conv_t_filters=[64, 64, 32, 3], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                       decoder_conv_t_strides=[2, 2, 2, 2], log_dir=args.logdir, z_dim=args.z_dim,
                                       kernel_visualization_layer=args.kernel_visualization_layer,
                                       feature_map_layers=args.feature_map_layers, use_batch_norm=args.use_batch_norm,
                                       decay_rate=args.lr_decay, num_samples=args.num_samples,
                                       feature_map_reduction_factor=args.feature_map_reduction_factor,
                                       inner_activation=args.inner_activation)
        data_gen = ImageDataGenerator(rescale=1. / 255)

        training_data = data_gen.flow_from_directory(os.path.join(args.data_path, dataset_subfolder),
                                                     target_size=INPUT_DIM[:2],
                                                     batch_size=args.batch_size,
                                                     shuffle=True, class_mode='input')
    elif args.configuration == 'celeba_large_model':
        from models.VAE import VariationalAutoencoder
        INPUT_DIM = (224, 224, 1)
        model = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[11, 7, 5, 3], encoder_conv_strides=[4, 2, 2, 2],
                                       decoder_conv_t_filters=[64, 64, 32, 1], decoder_conv_t_kernel_size=[3, 5, 7, 11],
                                       decoder_conv_t_strides=[2, 2, 2, 4], log_dir=args.logdir, z_dim=args.z_dim,
                                       use_batch_norm=args.use_batch_norm, use_dropout=args.use_dropout,
                                       kernel_visualization_layer=args.kernel_visualization_layer,
                                       num_samples=args.num_samples, dropout_rate=args.dropout_rate,
                                       inner_activation=args.inner_activation, decay_rate=args.lr_decay,
                                       feature_map_layers=args.feature_map_layers,
                                       feature_map_reduction_factor=args.feature_map_reduction_factor)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(os.path.join(args.data_path, dataset_subfolder),
                                                     target_size=INPUT_DIM[:2],
                                                     batch_size=args.batch_size,
                                                     shuffle=True, class_mode='input', interpolation='lanczos',
                                                     color_mode='grayscale')
    elif args.configuration == 'imagenet_classification_alexnet':
        INPUT_DIM = (224, 224, 3)
        model = AlexNet(input_dim=INPUT_DIM, log_dir=args.logdir, feature_map_layers=args.feature_map_layers,
                        use_batch_norm=args.use_batch_norm, decay_rate=args.lr_decay,
                        inner_activation=args.inner_activation,
                        kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                        use_fc=args.use_fc, feature_map_reduction_factor=args.feature_map_reduction_factor)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder),
            target_size=INPUT_DIM[:2], batch_size=args.batch_size,
            shuffle=True, class_mode='categorical',
            follow_links=True)
    elif args.configuration == 'alexnet_vae':
        INPUT_DIM = (224, 224, 3)
        model = AlexNetVAE(input_dim=INPUT_DIM, log_dir=args.logdir, z_dim=args.z_dim,
                           feature_map_layers=args.feature_map_layers, use_batch_norm=args.use_batch_norm,
                           kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                           use_fc=args.use_fc, inner_activation=args.inner_activation, decay_rate=args.lr_decay,
                           feature_map_reduction_factor=args.feature_map_reduction_factor)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder),
            target_size=INPUT_DIM[:2], batch_size=args.batch_size,
            class_mode='input', interpolation='lanczos',
            follow_links=True)
    elif args.configuration == 'alexnet_vae_classification_loss':
        INPUT_DIM = (224, 224, 3)
        model = AlexAlexNetVAE(input_dim=INPUT_DIM, log_dir=args.logdir, z_dim=args.z_dim,
                               feature_map_layers=args.feature_map_layers, use_batch_norm=args.use_batch_norm,
                               kernel_visualization_layer=args.kernel_visualization_layer, num_samples=args.num_samples,
                               use_fc=args.use_fc, inner_activation=args.inner_activation, decay_rate=args.lr_decay,
                               feature_map_reduction_factor=args.feature_map_reduction_factor,
                               alexnet_weights_path=args.alexnet_weights_path)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder),
            target_size=INPUT_DIM[:2], batch_size=args.batch_size,
            class_mode='input', interpolation='lanczos',
            follow_links=True)
    elif args.configuration == 'frozen_alexnet_vae':
        INPUT_DIM = (224, 224, 3)
        # TODO remove static requirements
        shape_before_flattening = (7, 7, 256)
        model = FrozenAlexNetVAE(z_dim=args.z_dim, use_dropout=args.use_dropout, dropout_rate=args.dropout_rate,
                                 use_batch_norm=args.use_batch_norm, shape_before_flattening=shape_before_flattening,
                                 input_dim=INPUT_DIM, log_dir=args.logdir, weights_path=args.alexnet_weights_path,
                                 kernel_visualization_layer=args.kernel_visualization_layer, decay_rate=args.lr_decay,
                                 inner_activation=args.inner_activation,
                                 feature_map_reduction_factor=args.feature_map_reduction_factor,
                                 feature_map_layers=args.feature_map_layers, num_samples=args.num_samples)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, dataset_subfolder),
            target_size=INPUT_DIM[:2], batch_size=args.batch_size,
            class_mode='input', interpolation='lanczos',
            follow_links=True)
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
        model.train(training_data, epochs=args.num_epochs, weights_folder=weights, batch_size=args.batch_size,
                    print_every_n_batches=args.print_every_n_batches, initial_epoch=args.initial_epoch,
                    training_labels=training_labels)
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


if __name__ == '__main__':
    main()
