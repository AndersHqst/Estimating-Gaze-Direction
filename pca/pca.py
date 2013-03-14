import numpy as np
import scipy.io
import matplotlib.pyplot as plt


# functions

def featureNormalize(data):
    ''' Normalizes data to a mean value of 0 and a standard deviation of 1 '''
    mean = np.mean(data, axis = 0)
    normalized = data - mean
    variance = np.std(normalized, axis = 0)
    normalized = normalized / variance
    return normalized, mean, variance

def getCovarianceMatrix(normalizedData):
    ''' Returns a covariance matrix, defined as (1/m)* xT * x (1 over m times x transposed x)
        where x has a row per sample and m is the number of samples '''
    m = normalizedData.shape[0]
    transposed = np.transpose(normalizedData)
    return transposed.dot(normalizedData) / m
    pass

# flow

data = scipy.io.loadmat('ex7data1.mat')
data = data['X']

normalizedData, mean, variance = featureNormalize(data)
covarianceMatrix = getCovarianceMatrix(normalizedData)
u, s, v = np.linalg.svd(covarianceMatrix)

plt.hold('on')
plt.plot(data[:,0], data[:,1], 'bo')
plt.figure

p1, p2 = mean, 1.5 * s[0] * np.transpose(u[:,0])
p3, p4 = mean, 1.5 * s[1] * np.transpose(u[:,1])
plt.arrow(p1[0], p1[1], p2[0], p2[1])
plt.arrow(p3[0], p3[1], p4[0], p4[1])


plt.axis([0.5, 6.5, 2, 8])
plt.show()


raw_input()





