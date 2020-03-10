from keras import backend as K
from keras.engine import Layer
from keras.layers import Dense, Lambda


def sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon


class NormalVariational(Layer):

    def __init__(self, size: int, name: str = None):
        super().__init__(name=name)
        self.mu_layer = Dense(size)
        self.sigma_layer = Dense(size)

    def call(self, inputs, **kwargs):
        mu = self.mu_layer(inputs)
        log_var = self.sigma_layer(inputs)
        return Lambda(sampling, name='encoder_output')([mu, log_var])
