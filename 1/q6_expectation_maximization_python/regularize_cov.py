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

    #####Insert your code here for subtask 6d#####
    covariances = np.zeros((2, 2, 3))
    covariances[:, :, 0] = np.identity(2)
    covariances[:, :, 1] = np.identity(2)
    covariances[:, :, 2] = np.identity(2)
    regularized_cov = covariance + epsilon * covariances

    return regularized_cov
