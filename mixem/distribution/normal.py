import numpy as np
import scipy.stats

from mixem.distribution.distribution import Distribution


class NormalDistribution(Distribution):
    """Univariate normal distribution with parameters (mu, sigma)."""

    @staticmethod
    def density(data, parameters):
        mu, sigma = parameters
        assert(len(data.shape) == 1), "Expect 1D data!"

        return scipy.stats.norm.pdf(data, mu, sigma)

    @staticmethod
    def estimate_parameters(data, weights):
        assert(len(data.shape) == 1), "Expect 1D data!"

        wsum = np.sum(weights)

        mu = np.sum(weights * data) / wsum
        sigma = np.sqrt(np.sum(weights * (data - mu) ** 2) / wsum)

        return mu, sigma


class MultivariateNormalDistribution(Distribution):
    """Multivariate normal distribution with parameters (mu, Sigma)."""

    @staticmethod
    def density(data, parameters):
        mu, sigma = parameters
        return scipy.stats.multivariate_normal.pdf(data, mu, sigma)

    @staticmethod
    def estimate_parameters(data, weights):
        mu = np.sum(data * weights[:, np.newaxis], axis=0) / np.sum(weights)

        center_x = data - mu[np.newaxis, :]

        # sigma = (np.diag(weights) @ center_x).T @ center_x / np.sum(weights)
        sigma = np.dot(
            np.dot(
                np.diag(weights),
                center_x
            ).T,
            center_x
        ) / np.sum(weights)

        return mu, sigma
