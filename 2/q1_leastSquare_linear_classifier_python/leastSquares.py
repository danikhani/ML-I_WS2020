import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #that trains a least-squares classifier based on a data matrix data and its class label vector label . It provides as output the linear classifier weight vector weight and bias.
    # page 29 lecture 6
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)

    # D = Number of attributes
    D = data.shape[0]

    ones = np.full(D,1)
    data = np.column_stack((ones,data))

    # Calculating the weights
    Weight = np.linalg.pinv(data)@label

    weight = Weight[1:]
    bias = Weight[0]

    return weight, bias
