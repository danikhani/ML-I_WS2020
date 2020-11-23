import numpy as np
import math as mth
from scipy.spatial import distance
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #log-likelihood ln p(X|π, μ, Σ) of a mixture of Gaussian distributions with the signature
    # s. 433
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    logLikelihood = 0
    D = X.shape[1] # 2 dimension set of files
    N = X.shape[0] # number of samples
    K = len(weights) # number of Gausians
    i = 0
    #book page 433 and 25 or slide part 5 slide 11
    while i < N:
        x_n = X[i, :]
        j = 0
        pz = 0
        while j < K:
            w = weights[j]
            c = covariances[:, :, j]
            m = means[j]
            # source https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html
            dd = distance.mahalanobis(x_n, m, c)
            pp = w*(1 / (pow(2 * mth.pi, D / 2) * pow(np.linalg.det(c), 0.5)) * mth.exp(-dd))
            pz = pz + pp
            j +=1
        logLikelihood = logLikelihood + np.log(pz)
        i += 1
    return logLikelihood

