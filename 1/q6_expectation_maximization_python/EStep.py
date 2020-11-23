import numpy as np
from scipy.spatial import distance
import math as mth
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #EStep that performs the expectation step of the EM algorithm, i.e. computes the responsibilities γ j (x n ):
    #S.338 part 5 slide 12
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    D = X.shape[1]  # 2 dimension set of files
    N = X.shape[0]  # number of samples
    K = len(weights) # number of Gausians

    # book page 433 and 25 or slide part 5 slide 11
    i =0
    gamma = np.zeros((N, 3))
    #gamma[3,2] = 25 # row 3 column 2 is 25
    #gamma[3,1] = 4
    #gamma [4,1]= 10
    #gamma[3,:] = gamma[3,:]/4
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
            pp = w * (1 / (pow(2 * mth.pi, D / 2) * pow(np.linalg.det(c), 0.5)) * mth.exp(-dd))
            pz = pz + pp
            #set the rows
            gamma[i,j] = pp
            j += 1
        gamma[i,:] = gamma[i,:]/pz
        i += 1

    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return [logLikelihood, gamma]
