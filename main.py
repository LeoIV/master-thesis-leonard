import os
from glob import glob
import numpy as np
import pdb
from keras import backend as K
from keras.datasets import mnist, cifar, cifar10

from models.VAE import VariationalAutoencoder
from keras.preprocessing.image import ImageDataGenerator

# run config
from utils.loaders import load_mnist

RUN_ID = '0001'
SECTION = 'VAE'
RUN_FOLDER = 'run/{}/'.format(SECTION)
DATA_NAME = 'faces'
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])
# data config
DATA_FOLDER = './data/celeb1/'
# model config
INPUT_DIM = (28, 28, 1)
BATCH_SIZE = 64
# train config
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 10000
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

if __name__ == '__main__':

    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        os.mkdir(os.path.join(RUN_FOLDER, 'images'))
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

    mode = 'build'  # 'load'
    filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
    num_images = len(filenames)

    (x_train, y_train), (x_test, y_test) = load_mnist()

    with open('logs/metadata.tsv', 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(y_train[:5000]):
            f.write("%d\t%d\n" % (index, int(label)))



    vae = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                 encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[1, 2, 2, 1],
                                 decoder_conv_t_filters=[64, 64, 32, 1], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                 decoder_conv_t_strides=[1, 2, 2, 1], z_dim=4)
    if mode == 'build':
        vae.save(RUN_FOLDER)
    else:
        vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

    vae.encoder.summary()
    vae.decoder.summary()

    vae.train(x_train, epochs=EPOCHS, run_folder=RUN_FOLDER, batch_size=BATCH_SIZE,
              print_every_n_batches=PRINT_EVERY_N_BATCHES, initial_epoch=INITIAL_EPOCH)
