import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #the function MStep that performs the maximization step of the EM algorithm. i.e. π̂j^new , μ̂j^new, Σ̂j^new :
    # s.439
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:

    nk = 1 / np.sum(gamma)

    # means          : Mean for each gaussian (KxD).
    means = 1/nk * sum(gamma*X)

    # weights        : Vector of weights of each gaussian (1xK).
    weights = nk/X.shape[0]
    # covariances    : Covariance matrices for each component(DxDxK).
    covariances = 1/nk * sum(gamma*X)
    # logLikelihood  : Log-likelihood (a scalar).
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood
