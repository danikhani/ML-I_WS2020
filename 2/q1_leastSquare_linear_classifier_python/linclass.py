import numpy as np

def linclass(weight, bias, data):
    # Linear Classifier
    # that classifies a data matrix data based on a trained linear classifier weight, bias .
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction+
    D = data.shape[0]
    K = data.shape[1]

    bias_vec = np.full(D,bias)

    class_pred = data@weight + bias_vec

    class_pred = np.sign(class_pred)
    return class_pred


