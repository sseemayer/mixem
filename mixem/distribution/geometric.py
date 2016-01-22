# coding=utf-8
import numpy as np

from mixem.distribution.distribution import Distribution


class GeometricDistribution(Distribution):
    """Geometric distribution with parameter (p)."""

    def __init__(self, p):
        self.p = p

    def density(self, data):
        assert(len(data.shape) == 1), "Expect 1D data!"
        return (1 - self.p) ** (data - 1) * self.p

    def estimate_parameters(self, data, weights):
        assert(len(data.shape) == 1), "Expect 1D data!"

        self.p = np.sum(weights) / np.sum(data * weights)

    def __repr__(self):
        return "Geom[p={p:.4g}]".format(p=self.p)
