import logging

import os
import sys
from shutil import rmtree

import traceback
from keras_preprocessing.image import ImageDataGenerator

from models.AlexNetVAE import AlexNetVAE
from models.VAE import VariationalAutoencoder
from models.AlexNet import AlexNet
# run config
from utils.loaders import load_mnist
from argparse import ArgumentParser


# create logger with 'spam_application'


def main():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='root.log', level=logging.DEBUG, filemode="w")

    parser = ArgumentParser(description='Functionality for Leonards master thesis')
    parser.add_argument('configuration', type=str,
                        choices=['mnist', 'celeba', 'celeba_large_model', 'imagenet_classification_alexnet',
                                 'imagenet_classification_alexnet_vae'],
                        help="The configuration to execute.\n\n"
                             "mnist: VAE trained on mnist\n"
                             "celeba: VAE trained on (128,128) sized celeba dataset\n"
                             "celeba_large_model: celeba trained on upscaled (224,224) sized images\n"
                             "imagenet_classification_alexnet: train AlexNet on imagenet classification task")
    parser.add_argument('data_path',
                        help="The path containing the individual datafolders. If the path to the imagenet folder is "
                             "/foo/bar/imagenet, the value should be /foo/bar/. Can be absolute or relative.")
    parser.add_argument('--logdir', type=str, default="logs/",
                        help="The directory to which the logs will be written. Absolute or relative.")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size.")
    parser.add_argument('--num_epochs', type=int, default=100, help="The number of epochs.")
    parser.add_argument('--initial_epoch', type=int, default=0, help="The initial epochs (0 for a new run).")
    parser.add_argument('--print_every_n_batches', type=int, default=250,
                        help="After how many batches the callbacks will be executed.")
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help="The initial learning rate passed to the optimizer.")
    parser.add_argument('--r_loss_factor', type=int, default=10E4,
                        help="The factor by which to weigh the reconstruction loss in case of variational autoencoder")
    parser.add_argument('--mode', type=str, choices=['build', 'load'], default='build',
                        help="Whether to build a new model or load an existing one.")
    parser.add_argument('--z_dim', type=int, default=200,
                        help="The embedding space dimensionality. Only considered for VAEs.")
    args = parser.parse_args()

    weights = os.path.join(args.logdir, 'weights')

    if not os.path.exists(args.logdir):
        logging.info("Creating logdir: {}".format(args.logdir))
        os.mkdir(args.logdir)
    if not os.path.exists(weights):
        logging.info("Creating weights dir: {}".format(weights))
        os.mkdir(weights)

    if not len(os.listdir(args.logdir)) == 0:
        logging.warning("Logdir not empty. Deleting content...")
        rmtree(args.logdir)
        os.mkdir(args.logdir)

    if args.configuration == 'mnist':
        INPUT_DIM = (28, 28, 1)
        model = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[1, 2, 2, 1],
                                       decoder_conv_t_filters=[64, 64, 32, 1], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                       decoder_conv_t_strides=[1, 2, 2, 1], log_dir=args.logdir, z_dim=args.z_dim)
        (training_data, _), (_, _) = load_mnist()

    elif args.configuration == 'celeba':
        INPUT_DIM = (128, 128, 3)
        model = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[2, 2, 2, 2],
                                       decoder_conv_t_filters=[64, 64, 32, 3], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                       decoder_conv_t_strides=[2, 2, 2, 2], log_dir=args.logdir, z_dim=args.z_dim)
        data_gen = ImageDataGenerator(rescale=1. / 255)

        training_data = data_gen.flow_from_directory(os.path.join(args.data_path, 'celeb/'), target_size=INPUT_DIM[:2],
                                                     batch_size=args.batch_size,
                                                     shuffle=True, class_mode='input')
    elif args.configuration == 'celeba_large_model':
        INPUT_DIM = (224, 224, 1)
        model = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                       encoder_conv_kernel_size=[11, 7, 5, 3], encoder_conv_strides=[4, 2, 2, 2],
                                       decoder_conv_t_filters=[64, 64, 32, 1], decoder_conv_t_kernel_size=[3, 5, 7, 11],
                                       decoder_conv_t_strides=[2, 2, 2, 4], log_dir=args.logdir, z_dim=args.z_dim)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(os.path.join(args.data_path, 'celeb/'), target_size=INPUT_DIM[:2],
                                                     batch_size=args.batch_size,
                                                     shuffle=True, class_mode='input', interpolation='lanczos',
                                                     color_mode='grayscale')
    elif args.configuration == 'imagenet_classification_alexnet':
        INPUT_DIM = (224, 224, 3)
        model = AlexNet(input_dim=INPUT_DIM, log_dir=args.logdir)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, 'imagenet/ILSVRC/Data/CLS-LOC/train/'),
            target_size=INPUT_DIM[:2], batch_size=args.batch_size,
            shuffle=True, class_mode='categorical',
            follow_links=True)
    elif args.configuration == 'imagenet_classification_alexnet_vae':
        INPUT_DIM = (224, 224, 3)
        model = AlexNetVAE(input_dim=INPUT_DIM, log_dir=args.logdir, z_dim=args.z_dim)
        data_gen = ImageDataGenerator(rescale=1. / 255)
        training_data = data_gen.flow_from_directory(
            directory=os.path.join(args.data_path, 'imagenet/ILSVRC/Data/CLS-LOC/train/'),
            target_size=INPUT_DIM[:2], batch_size=args.batch_size,
            class_mode='input', interpolation='lanczos',
            follow_links=True)
    if args.mode == 'build':
        model.save(weights)
    else:
        model.load_weights(os.path.join(weights, 'weights/weights.h5'))

    model.compile(args.learning_rate, args.r_loss_factor)

    model.model.summary()
    try:
        model.train(training_data, epochs=args.num_epochs, run_folder=weights, batch_size=args.batch_size,
                    print_every_n_batches=args.print_every_n_batches, initial_epoch=args.initial_epoch)
    except Exception as e:
        logging.error("An error occurred during training:")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
            logging.error(line)
        raise e


if __name__ == '__main__':
    main()
