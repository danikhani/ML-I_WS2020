import numpy as np
import scipy

def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]
    pos = np.arange(-5, 5.0, 0.1)  # Returns a 1 dimensional vector with 100 enetries

    n = samples.shape[0]

    temp_value = []
    distance_list = {}

    for x in pos:
        distance_from_one_point =[]
        for j in samples:
            distance = abs(x - j)
            distance_from_one_point.append(distance)
        distance_from_one_point.sort(reverse=0)
        distance_list.update({x:distance_from_one_point})

        p_x = k/(n*distance_from_one_point[k+1])
        temp_value.append(p_x)

        #p_x = k/(n*v)
        #temp_value.append(p_x)

    print('heeelp')
    value = np.array(temp_value)
    estDensity = np.column_stack((pos, value))
    return estDensity
