import numpy as np
import scipy.stats

from mixem.distribution.distribution import Distribution


class NormalDistribution(Distribution):
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
