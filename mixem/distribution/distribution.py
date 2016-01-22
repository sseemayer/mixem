import abc


class Distribution(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def density(self, data, parameters):
        raise NotImplementedError("Need to implement density calculation!")

    @abc.abstractmethod
    def estimate_parameters(self, data, weights):
        raise NotImplementedError("Need to implement parameter estimation!")

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError("Need to implement string representation!")
