import numpy as np
from keras import backend as K, Model
from keras.engine import Layer
from keras.layers import Dense, Lambda


def distance_measure(y1, y2, perc_loss: Model):
    feat_1 = perc_loss.predict(y1)
    feat_2 = perc_loss.predict(y2)
    squared_difference = np.square(feat_1 - feat_2)
    return np.mean(squared_difference, axis=(1, 2, 3))


def normalize(v):
    im = np.sqrt(np.sum(np.square(v), axis=-1, keepdims=True))
    return v / im


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = np.sum(a * b, axis=-1, keepdims=True)
    p = t * np.arccos(d)
    c = normalize(b - d * a)
    d = a * np.cos(p) + c * np.sin(p)
    return normalize(d)


def perceptual_path_length(decoder, perc_loss: Model, num_samples=100, minibatch=32, sampling='full', epsilon=1e-4,
                           seed=None):
    distance_expr = []
    for begin in range(0, num_samples, minibatch):
        if seed is not None:
            np.random.seed(begin * seed)
        lat_t01 = np.random.standard_normal([minibatch * 2] + decoder.inputs[0].shape[1:])
        lerp_t = np.random.uniform(size=[minibatch], low=0.0, high=1.0 if sampling == 'full' else 0.0)
        lat_t0, lat_t1 = lat_t01[0::2], lat_t01[1::2]
        lat_e0 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis])
        lat_e1 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis] + epsilon)
        dlat_e01 = np.reshape(np.stack([lat_e0, lat_e1], axis=1), lat_t01.shape)
        images = decoder.predict(dlat_e01)

        img_e0, img_e1 = images[0::2], images[1::2]

        distance_expr.append(distance_measure(img_e0, img_e1, perc_loss) * (1 / epsilon ** 2))
    all_distances = np.concatenate(distance_expr, axis=0)
    return np.mean(all_distances), all_distances


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
