# -*- coding: utf-8 -*-
from mixem.distribution import Distribution
import numpy as np
import scipy.special

def l2norm(x,axis=None):
    res = np.sum(x**2,axis=axis)**0.5
    return res

# class vmfDistribution(mixem.distribution.Distribution):
class vmfDistribution(Distribution):
    """Von-mises Fisher distribution with parameters (mu, kappa).
    Ref: Clustering on the Unit Hypersphere using von Mises-Fisher Distributions
    http://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
    """

    def __init__(self, mu, 
                 kappa = None,
                ):
        mu = np.array(mu)
        
        assert len(mu.shape) == 1, "Expect mu to be 1D vector!"
        if all(mu==0):
            self.dummy = True
        else:
            self.dummy = False
            
        if kappa is not None:
            assert len(np.shape(kappa)) == 0,"Expect kappa to be 0D vector"
            kappa = float(kappa)
        self.kappa = kappa

        self.mu = mu
        self.radius = np.sum(self.mu ** 2) ** 0.5
        self.D = len(mu)

    def log_density(self, data):
#         L2 = np.sum(np.square(data),axis=1,keepdims=1)
#         return  np.dot(data, self.mu) * L2 / L2
        
        logP = np.dot(data, self.mu) 
        if self.kappa is not None:
            normTerm = ( - np.log(scipy.special.iv(self.D/2. -1.,  self.kappa ))
                          + np.log(self.kappa) * (self.D/2. - 1.)
                          - np.log(2*np.pi) * self.D/2.
                        ) 
            logP = logP * self.kappa + normTerm
        return  logP

    def estimate_parameters(self, data, weights):
        if not self.dummy:
            L2 = np.sum(np.square(data),axis=1,keepdims=1)
            L2sqrt = np.sqrt(L2)
#             fct = np.exp(L2sqrt)/L2sqrt
#             fct = np.exp(L2sqrt)
            fct = 1.
            wdata = data * fct
        
            wwdata  = wdata * weights[:, np.newaxis]
            rvct = np.sum(wwdata, axis=0) / np.sum(weights)
            rnorm = l2norm(rvct)
            self.mu = rvct / rnorm * self.radius
            
            if self.kappa is not None:
                r = rnorm
                self.kappa = (r * self.D - r **3 )/(1. - r **2)
            

    def __repr__(self):
        po = np.get_printoptions()

        np.set_printoptions(precision=3)

        try:
            result = "MultiNorm[μ={mu},]".format(mu=self.mu,)
        finally:
            np.set_printoptions(**po)

        return result