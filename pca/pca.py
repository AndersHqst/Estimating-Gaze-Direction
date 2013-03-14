import numpy as np
import scipy.io
import matplotlib.pyplot as plt


# functions ----

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

def plotData(data, u, s, v):
    plt.hold('on')
    plt.plot(data[:,0], data[:,1], 'bo')
    
    p1, p2 = mean, 1.5 * s[0] * np.transpose(u[:,0]) # ved ikke lige hvad de 1.5 laver, antager det er for plottets skyld
    p3, p4 = mean, 1.5 * s[1] * np.transpose(u[:,1])
    plt.arrow(p1[0], p1[1], p2[0], p2[1])
    plt.arrow(p3[0], p3[1], p4[0], p4[1])
    
    plt.axis([0.5, 6.5, 2, 8])
    plt.show()


# flow ----

data = scipy.io.loadmat('ex7data1.mat')
data = data['X']

normalizedData, mean, variance = featureNormalize(data)
covarianceMatrix = getCovarianceMatrix(normalizedData)

u, s, v = np.linalg.svd(covarianceMatrix) # numpy giver et s-array hvor kun diagonalerne fra S-matrixen beskrevet i kurset er med

plotData(data, u, s, v)


raw_input()





