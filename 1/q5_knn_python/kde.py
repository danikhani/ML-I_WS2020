import numpy as np
import math as mth

#kernel density estimation method with a Gaussian kernel with standard deviation h (Seite 123 Bishop buch)
def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]
    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector

    n = samples.shape[0]
    #d = pos.shape[0]
    d = 1

    temp_value = []
    for x in pos:
        summe = 0
        for j in range(0, n-1): # summe von 1 bis n
            diff_norm = np.linalg.norm(x - samples[j])
            summe += 1/(pow(2*mth.pi*pow(h,2),d/2))*mth.exp(-1*pow(diff_norm,2)/(2*pow(h,2)))
        p_x = 1/n * summe
        temp_value.append(p_x)

    value = np.array(temp_value)
    estDensity = np.column_stack((pos, value))

    return estDensity
