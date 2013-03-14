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

def plotOriginalData(data, u, s, v):
    plt.hold('on')
    plt.plot(data[:,0], data[:,1], 'bo')
    
    p1, p2 = mean, 1.5 * s[0] * np.transpose(u[:,0]) # ved ikke lige hvad de 1.5 laver, antager det er for plottets skyld
    p3, p4 = mean, 1.5 * s[1] * np.transpose(u[:,1])
    plt.arrow(p1[0], p1[1], p2[0], p2[1])
    plt.arrow(p3[0], p3[1], p4[0], p4[1])
    
    plt.axis([0.5, 6.5, 2, 8])
    plt.show()
    plt.hold('off')

def projectData(normalizedData, u, maxK):
    projected = np.zeros((normalizedData.shape[0], maxK))
    m = normalizedData.shape[0]
    for k in range(maxK):
        for i in range(m):
            sample = normalizedData[i, :]
            projected[i, k] = sample.dot(u[:, k])
            
    return projected

def recoverData(projectedData, u, maxK):         
    m = projectedData.shape[0]
    dimensions = u.shape[0]
    recovered = np.zeros((m, dimensions))
    
    for j in range(dimensions):
        for i in range(m):
            projectedSample = projectedData[i, :]
            recovered[i, j] = projectedSample.dot(np.transpose(u[j, 0:maxK]))

    return recovered


def plotRecoveredData(recovered, normalized):
    plt.hold('on')
    plt.plot(recovered[:, 0], recovered[:, 1], 'ro')
    diff = recovered - normalized
    for i in range(recovered.shape[0]):
        plt.arrow(normalized[i,0], normalized[i,1], diff[i,0], diff[i,1])
    
    plt.show()
    plt.hold('off')
    

    

# flow ----

data = scipy.io.loadmat('ex7data1.mat')
data = data['X']

normalizedData, mean, variance = featureNormalize(data)
covarianceMatrix = getCovarianceMatrix(normalizedData)

u, s, v = np.linalg.svd(covarianceMatrix) # numpy giver et s-array hvor kun diagonalerne fra S-matrixen beskrevet i kurset er med

# plotData(data, u, s, v)

projectedData = projectData(normalizedData, u, maxK = 1)
recoveredData = recoverData(projectedData, u, maxK = 1)

plotRecoveredData(recoveredData, normalizedData)



raw_input()





