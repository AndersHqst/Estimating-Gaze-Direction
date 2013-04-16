import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os
from eyeVideoLoader import EyeVideoLoader
import sys
from mpl_toolkits.mplot3d import Axes3D


class SliderHandler:

    def __init__(self, face, mean, variance, u, imageSize, maxK = 100):
        self.face = face
        self.mean = mean
        self.variance = variance
        self.imageSize = imageSize
        self.maxK = maxK
        self.u = u

        cv2.namedWindow("Sliders", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Face")

        for i in range(maxK):
            cv2.createTrackbar(str(i), "Sliders", int(self.face[i]) + 70, 140, self.updateFace)
        self.updateFace()

    def updateFace(self, dummy = None):
        for i in range(self.maxK):
            sliderValue = cv2.getTrackbarPos(str(i), "Sliders")
            self.face[i] = sliderValue - 70

        recoveredFace = recoverData(self.face, self.u, maxK = self.maxK)
        recoveredFace = deNormalize(recoveredFace, self.mean, self.variance)
        cv2.normalize(recoveredFace, recoveredFace, 0, 255, cv2.NORM_MINMAX)
        recoveredFace = recoveredFace.astype('uint8').reshape(self.imageSize)#.transpose()

        recoveredFace = cv2.pyrUp(recoveredFace)
        cv2.imshow("Face", recoveredFace)


def loadData1():
    data = scipy.io.loadmat('ex7data1.mat')
    return data['X']


def loadFaceData():
    data = scipy.io.loadmat('ex7faces.mat')
    return data['X']



def featureNormalize(data):
    ''' Normalizes each feature (column) of the data to a mean value of 0 and a standard deviation of 1 '''
    return normalize(data, axis = 0)


def sampleNormalize(data):
    return normalize(data, axis = 1)


# We do not need to divide by the standard deviation. This is done for the sake
# of feature scaling, and is only relevant when varibles in our data are 'different'
# For images, all variables have the same scale.
# Andrew Ng explains it here: http://www.youtube.com/watch?v=ey2PE5xi9-A?t=43m
# (couldn't get it to work without it so I'll leave it in for now)

def normalize(data, axis):
    mean = np.mean(data, axis = axis) #.reshape(-1, 1)
    normalized = data - mean
    variance = np.std(normalized, axis = axis) #.reshape(-1, 1)
    normalized = normalized / variance
    return normalized, mean, variance


def deNormalize(normalizedData, mean, variance):
    return normalizedData * variance + mean


def getCovarianceMatrix(normalizedData):
    ''' Returns a covariance matrix, defined as (1/m)* xT * x (1 over m times x transposed x)
        where x has a row per sample and m is the number of samples '''
    m = normalizedData.shape[0]
    transposed = np.transpose(normalizedData)
    return transposed.dot(normalizedData) / m

def kDimensionWithVaraianceRetained(data, variance):
    '''Return the minimum k dimensions needed for retaining variance
    :param variance: eg 0.99 for 99% variance retained
    :param data: data
    '''
    k = 1
    vars = []
    normalized, mean, var = featureNormalize(data)
    covMat = getCovarianceMatrix(normalized)
    (u, s, v) = np.linalg.svd(covMat)
    while k < len(data[0]):
        val = sum(s[:k]) / sum(s[:len(data)])
        vars.append(val)
        # print 'Variance calculated: %s With k: %s' % (val, k)
        if val >= variance:
            break
        k += 1
    plt.plot([x for x in range(len(vars))], vars, 'b-')    
    plt.xlabel('k')
    plt.ylabel('Variance retained')
    plt.show()
    return k


def plotOriginalData(data, u, s, v, mean):
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
    u = u[:, 0:maxK]
    #In the Stanford video this is calculated as u transpose x (column vector).
    #The return below is equalivant because our data
    #has a feature per column, and not per row (row vectors).
    #Thus, to do the same as in the video, we would have to write
    #u transpose dot x transpose == x dot u
    return normalizedData.dot(u)

def recoverData(projectedData, u, maxK):
    u = np.transpose(u[:, 0:maxK])
    return projectedData.dot(u)


def plotRecoveredData(recovered, normalized):
    plt.hold('on')
    plt.plot(recovered[:, 0], recovered[:, 1], 'ro')
    diff = recovered - normalized
    for i in range(recovered.shape[0]):
        plt.arrow(normalized[i,0], normalized[i,1], diff[i,0], diff[i,1])
    
    plt.show()
    plt.hold('off')
    

def runPart1():
    ''' 2.3: 2D to 1D '''
    data = loadData1()
    (normalizedData, mean, variance) = featureNormalize(data)
    
    covarianceMatrix = getCovarianceMatrix(normalizedData)
    
    (u, s, v) = np.linalg.svd(covarianceMatrix) # numpy giver et s-array hvor kun diagonalerne fra S-matrixen beskrevet i kurset er med
    plotOriginalData(data, u, s, v, mean)
    
    projectedData = projectData(normalizedData, u, maxK = 1)
    recoveredData = recoverData(projectedData, u, maxK = 1)
    plotRecoveredData(recoveredData, normalizedData)
    

def show100Faces(faces, size):
    plt.gray()
    display = None
    row = None
    index = 0
    for r in range(10):
        for c in range(10):
            face = faces[index].reshape(size) # .transpose()
            index += 1
            if (row is None):
                row = face
            else:
                row = np.concatenate((row, face), axis = 1)
        
        if display is None:
            display = row
        else:
            display = np.concatenate((display, row), axis = 0)
        row = None
        
    plt.imshow(display)
    plt.show()


def runPart2():
    ''' 2.4: Faces '''

    faces = loadFaceData()
    show100Faces(faces)

    (normalizedFaces, mean, variance) = featureNormalize(faces)
    covarianceMatrix = getCovarianceMatrix(normalizedFaces)
    (u, s, v) = np.linalg.svd(covarianceMatrix)
    show100Faces(u.transpose())

    projectedFaces = projectData(normalizedFaces, u, maxK = 100)
    recoveredFaces = recoverData(projectedFaces, u, maxK = 100)
    recoveredFaces = deNormalize(recoveredFaces, mean, variance)

    show100Faces(recoveredFaces)

    sliderFace = projectedFaces[0]
    sliderHandler = SliderHandler(sliderFace, mean, variance, u, (32,32), 100)

    while True:
        cv2.waitKey(10)


def plotProjectedData2D(data, targets):
    params = ('bo', 'ro', 'go', 'yo')
    for target in range(4):
        ii = np.nonzero(targets == target + 1)[0]
        plt.plot(projectedData[ii,0].flatten(),
                 projectedData[ii,1].flatten(), 
                 params[target])
    plt.show()


def plotProjectedData3D(data, targets):
    params = ('bo', 'ro', 'go', 'yo')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for target in range(4):
        ii = np.nonzero(targets == target + 1)[0]
        ax.plot(projectedData[ii,0].flatten(),
                projectedData[ii,1].flatten(),
                projectedData[ii,2].flatten(),
                params[target])
    plt.show()



#runPart1()
#runPart2()


loader = EyeVideoLoader()

#loader.normalizeSampleImages()
#sys.exit()

# loader.resizeEyeVideos()

# (eyeData, targets) = loader.loadDataFromVideos()

#np.save('eyeData.npy', eyeData)
#np.save('targets.npy', targets)
eyeData = np.load('eyeData.npy')
targets = np.load('targets.npy')


kDimensionWithVaraianceRetained(eyeData, 0.99)
#
interval = int(eyeData.shape[0] / 100)
show100Faces(eyeData[::interval], (28,42))

normalizedData, mean, variance = featureNormalize(eyeData) #sampleNormalize(eyeData) # 

covarianceMatrix = getCovarianceMatrix(normalizedData)
(u, s, v) = np.linalg.svd(covarianceMatrix)

show100Faces(u.transpose(), (28,42))


k = 20
projectedData = projectData(normalizedData, u, maxK = k)
plotProjectedData2D(projectedData, targets)

recoveredData = recoverData(projectedData, u, maxK = k)
recoveredData = deNormalize(recoveredData, mean, variance)

show100Faces(recoveredData[::interval], (28,42))

sliderEye = projectedData[0]
sliderHandler = SliderHandler(sliderEye, mean, variance, u, (28,42), maxK = k)

while True:
    cv2.waitKey(10)


