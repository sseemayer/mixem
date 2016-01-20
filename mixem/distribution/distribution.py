import abc
import numpy as np


class Distribution(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def density(self, data, parameters):
        return np.ones((data.shape[0],))

    @abc.abstractmethod
    def estimate_parameters(self, data, weights):
        return [], 0
