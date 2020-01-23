import os

from shutil import rmtree

from keras_preprocessing.image import ImageDataGenerator

from models.VAE import VariationalAutoencoder
# run config
from utils.loaders import load_mnist

# model config
BATCH_SIZE = 32
# train config
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 10000
EPOCHS = 10
PRINT_EVERY_N_BATCHES = 250
INITIAL_EPOCH = 0
LOGDIR = 'logs/'
WEIGHTS = 'weights'
DATASET = 'mnist'

if __name__ == '__main__':

    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    if not os.path.exists(WEIGHTS):
        os.mkdir(WEIGHTS)
    if not len(os.listdir(LOGDIR)) == 0:
        print("WARNING: Logdir not empty. Deleting content...")
        rmtree(LOGDIR)
        os.mkdir(LOGDIR)

    mode = 'build'  # 'load'

    if DATASET == 'mnist':
        INPUT_DIM = (28, 28, 1)
        vae = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                     encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[1, 2, 2, 1],
                                     decoder_conv_t_filters=[64, 64, 32, 1], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                     decoder_conv_t_strides=[1, 2, 2, 1], log_dir='./logs/', z_dim=4)
        (training_data, _), (_, _) = load_mnist()

    elif DATASET == 'celeba':
        INPUT_DIM = (128, 128, 3)
        vae = VariationalAutoencoder(input_dim=INPUT_DIM, encoder_conv_filters=[32, 64, 64, 64],
                                     encoder_conv_kernel_size=[3, 3, 3, 3], encoder_conv_strides=[2, 2, 2, 2],
                                     decoder_conv_t_filters=[64, 64, 32, 3], decoder_conv_t_kernel_size=[3, 3, 3, 3],
                                     decoder_conv_t_strides=[2, 2, 2, 2], log_dir='./logs/', z_dim=200)
        data_gen = ImageDataGenerator(rescale=1. / 255)

        training_data = data_gen.flow_from_directory('./data/celeb/', target_size=INPUT_DIM[:2], batch_size=BATCH_SIZE,
                                                     shuffle=True, class_mode='input')

    if mode == 'build':
        vae.save(WEIGHTS)
    else:
        vae.load_weights(os.path.join(WEIGHTS, 'weights/weights.h5'))

    vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

    vae.encoder.summary()
    vae.decoder.summary()

    vae.train(training_data, epochs=EPOCHS, run_folder=WEIGHTS, batch_size=BATCH_SIZE,
              print_every_n_batches=PRINT_EVERY_N_BATCHES, initial_epoch=INITIAL_EPOCH)
