import numpy as np


def cross_cumulants(data: np.ndarray):
    squared_cumulants = []
    data = data.reshape((data.shape[0], -1))

    # center data
    mean = data.mean(axis=0)
    data = data - np.expand_dims(mean, axis=0)
    # whiten data
    # data = whiten(data)

    len_data = data.shape[1]
    for index in np.ndindex(*([len_data] * 4)):
        # all indices equal
        if len(set(index)) == 1:
            continue
        squared_cumulants.append(np.square(_cumulants(data, *index)))
    return np.mean(squared_cumulants)


def _dot(*arrs):
    arr = np.array(arrs)
    return np.sum([np.prod(k) for k in arr.T])


def _cumulants(data: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """
    Compute the cumulants of data w.r.t to the component indices given by i, j, k, and l. Data will be centered.
    :param data: the signal, axis 0 is assumed to contain different datapoints, other axes are flattened. data has to be centered and whitened.
    :param i: first index
    :param j: second index
    :param k: third index
    :param l: fourth index
    :return: cumulants of data w.r.t the indices
    """
    assert len(data.shape) == 2

    ijkl = np.mean(np.prod(data[:, [i, j, k, l]], axis=1))
    ij = np.mean(np.prod(data[:, [i, j]], axis=1))
    kl = np.mean(np.prod(data[:, [k, l]], axis=1))
    ik = np.mean(np.prod(data[:, [i, k]], axis=1))
    jl = np.mean(np.prod(data[:, [j, l]], axis=1))
    il = np.mean(np.prod(data[:, [i, l]], axis=1))
    jk = np.mean(np.prod(data[:, [j, k]], axis=1))

    return ijkl - ij * kl - ik * jl - il * jk
