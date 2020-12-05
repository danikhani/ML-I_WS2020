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
    D = X.shape[1]  # 2 dimension set of files
    N = X.shape[0]  # number of samples
    K = gamma.shape[1]  # number of Gausians

    nk = gamma.sum(axis=0)

    # means          : Mean for each gaussian (KxD).
    means = np.zeros((K, D))
    means[0, :] = np.dot(gamma[:,0],X)/nk[0]
    means[1, :] = np.dot(gamma[:, 1], X)/nk[1]
    means[2, :] = np.dot(gamma[:, 2], X)/nk[2]

    # weights        : Vector of weights of each gaussian (1xK).
    weights = nk/nk.sum()





    # covariances    : Covariance matrices for each component(DxDxK).
    covariances = np.zeros((D, D, K))
    sdsd = np.size(covariances, 2)
    j = 0
    while j < K:
        i = 0
        sum = 0
        while i < 3:
            sum += gamma[i, j] * np.outer((X[i] - means[j]), (X[i] - means[j]))  # Transpose? Dimensions?
            i += 1
        covariances[:, :, j] = sum * 1 / nk[j]
        j += 1

    # logLikelihood  : Log-likelihood (a scalar).
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return weights, means, covariances, logLikelihood
