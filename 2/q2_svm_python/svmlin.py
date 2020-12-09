import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
#import os
#os.environ["PATH"] += os.pathsep + ...
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)


    #####Insert your code here for subtask 2a#####
    m, n = X.shape
    y = t.reshape(-1, 1) * 1.
    X_dash = y * X
    H = np.dot(X_dash, X_dash.T) * 1.

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))




    #n = X.shape[0]
    #q = (-1)* np.ones(n)
    #G = np.vstack([-np.eye(n), np.eye(n)])
    #A = t
    #b = 0
    #h = np.hstack([0,C])
    #P = np.full((n,n),0)
    #for i in range(n):
    #    for j in range(n):
    #        P[i,j] = t[i]*t[j]*np.dot(X[i],X[j])

    a= cvxopt_solvers.qp(P,q,G,h,A,b)
    alphas = np.array(sol['x'])
    # w parameter in vectorized form
    w = ((y * alphas).T @ X).reshape(-1, 1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()

    # Computing b
    b = y[S] - np.dot(X[S], w)

    alpha, sv, w, b, result, slack = svmlin(train['data'],train['label'],C)
    return alpha, sv, w, b, result, slack
