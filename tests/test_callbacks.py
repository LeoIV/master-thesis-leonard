from utils.callbacks import _correlation_coefficient
import numpy as np


def test_correlation_coefficient():
    a = [1, 0]
    b = [1, 0]
    assert _correlation_coefficient(a, b) == 1
    a, b = np.random.multivariate_normal((0, 0), np.zeros((2, 2)), 2)
    print(a)
    print(b)
    assert _correlation_coefficient(a, b) == 0
