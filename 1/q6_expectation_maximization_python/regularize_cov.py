import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix
    K = np.size(covariance, 2)
    D = np.size(covariance, 1)

    #####Insert your code here for subtask 6d#####
    covariances = np.zeros((D, D, K))
    for i in range(K):
        covariances[:, :, i] = np.identity(2)

    regularized_cov = covariance + epsilon * covariances

    return regularized_cov
