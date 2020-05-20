from keras import backend as K
from keras.engine import Layer
from keras.layers import Dense, Lambda


def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon


def precision_weighted_sampler(mu_logvar_1, mu_logvar_2):
    # assume input Tensors are (BATCH_SIZE, dime)
    mu1, sigma1 = mu_logvar_1
    mu2, sigma2 = mu_logvar_2
    size_1 = K.int_shape(mu1)
    size_2 = K.int_shape(mu2)

    if size_1 > size_2:
        print('convert 1d to 1d:', size_2, '->', size_1)
        mu2 = Dense(size_1)(mu2)
        sigma2 = Dense(size_1)(sigma2)
        mu_logvar_2 = (mu2, sigma2)
    elif size_1 < size_2:
        raise ValueError("musigma1 must be equal or bigger than musigma2.")

    mu, logsigma, sigma = precision_weighted(mu_logvar_1, mu_logvar_2)
    return mu + sigma * K.truncated_normal(K.int_shape(mu)), mu, logsigma


def precision_weighted(mu_logvar_1, mu_logvar_2):
    mu1, logvar1 = mu_logvar_1
    mu2, logvar2 = mu_logvar_2
    sigma1, sigma2 = K.exp(logvar1), K.exp(logvar2)
    sigma1__2 = 1 / K.square(sigma1)
    sigma2__2 = 1 / K.square(sigma2)
    mu = (mu1 * sigma1__2 + mu2 * sigma2__2) / (sigma1__2 + sigma2__2)
    sigma = 1 / (sigma1__2 + sigma2__2)
    logsigma = K.log(sigma + 1e-8)
    return mu, logsigma, sigma


class NormalVariational(Layer):

    def __init__(self, size: int, name: str = None):
        super().__init__(name=name)
        self.mu_layer = Dense(size)
        self.sigma_layer = Dense(size)

    def call(self, inputs, **kwargs):
        mu = self.mu_layer(inputs)
        log_var = self.sigma_layer(inputs)
        return Lambda(sampling, name='encoder_output')([mu, log_var])
