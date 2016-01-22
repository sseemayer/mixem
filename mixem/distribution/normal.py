# coding=utf-8
import numpy as np
import scipy.stats

from mixem.distribution.distribution import Distribution


class NormalDistribution(Distribution):
    """Univariate normal distribution with parameters (mu, sigma)."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def density(self, data):
        assert(len(data.shape) == 1), "Expect 1D data!"

        return scipy.stats.norm.pdf(data, self.mu, self.sigma)

    def estimate_parameters(self, data, weights):
        assert(len(data.shape) == 1), "Expect 1D data!"

        wsum = np.sum(weights)

        self.mu = np.sum(weights * data) / wsum
        self.sigma = np.sqrt(np.sum(weights * (data - self.mu) ** 2) / wsum)

    def __repr__(self):
        return "Normal[μ={mu:.4g}, σ={sigma:.4g}]".format(mu=self.mu, sigma=self.sigma)


class MultivariateNormalDistribution(Distribution):
    """Multivariate normal distribution with parameters (mu, Sigma)."""

    def __init__(self, mu, sigma):
        mu = np.array(mu)
        sigma = np.array(sigma)

        assert len(mu.shape) == 1, "Expect mu to be 1D vector!"
        assert len(sigma.shape) == 2, "Expect sigma to be 2D matrix!"

        assert sigma.shape[0] == sigma.shape[1], "Expect sigma to be a square matrix!"

        self.mu = mu
        self.sigma = sigma

    def density(self, data):
        return scipy.stats.multivariate_normal.pdf(data, self.mu, self.sigma)

    def estimate_parameters(self, data, weights):
        self.mu = np.sum(data * weights[:, np.newaxis], axis=0) / np.sum(weights)

        center_x = data - self.mu[np.newaxis, :]

        # sigma = (np.diag(weights) @ center_x).T @ center_x / np.sum(weights)
        self.sigma = np.dot(
            np.dot(
                np.diag(weights),
                center_x
            ).T,
            center_x
        ) / np.sum(weights)

    def __repr__(self):
        return "MultiNormal[μ={mu}, σ={sigma}]".format(mu=self.mu, sigma=self.sigma)

