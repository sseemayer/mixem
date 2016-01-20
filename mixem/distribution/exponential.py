import numpy as np

from mixem.distribution.distribution import Distribution


class ExponentialDistribution(Distribution):
    """Exponential distribution with parameter (lambda)."""

    @staticmethod
    def density(data, lmbda):
        assert(len(data.shape) == 1), "Expect 1D data!"
        return lmbda * np.exp(-lmbda * data)

    @staticmethod
    def estimate_parameters(data, weights):
        assert(len(data.shape) == 1), "Expect 1D data!"
        lmbda = np.sum(weights) / np.sum(weights * data)
        return lmbda
