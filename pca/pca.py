import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import os
from eyeVideoLoader import EyeVideoLoader
import os


class SliderHandler:

    def __init__(self, face, mean, variance, imageSize):
        self.face = face
        self.mean = mean
        self.variance = variance
        self.imageSize = imageSize

        cv2.namedWindow("Sliders", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Face")

        for i in range(20):
            cv2.createTrackbar(str(i+1), "Sliders", int(self.face[i]) + 70, 140, self.updateFace)
        self.updateFace()

    def updateFace(self, dummy = None):
        for i in range(20):
            sliderValue = cv2.getTrackbarPos(str(i), "Sliders")
            self.face[i] = sliderValue - 70

        recoveredFace = recoverData(self.face, u, maxK = 100)
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
    mean = np.mean(data, axis = 0)
    normalized = data - mean
    variance = np.std(normalized, axis = 0)
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
    

def show100Faces(faces):
    plt.gray()
    display = None
    row = None
    index = 0
    for r in range(10):
        for c in range(10):
            face = faces[index].reshape(32,32).transpose()
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
    sliderHandler = SliderHandler(sliderFace, mean, variance)

    while True:
        cv2.waitKey(10)


#runPart1()
#runPart2()


loader = EyeVideoLoader()

# loader.resizeEyeVideos()

(eyeData, targets) = loader.loadDataFromVideos()

normalizedData, mean, variance = featureNormalize(eyeData)
covarianceMatrix = getCovarianceMatrix(normalizedData)
(u, s, v) = np.linalg.svd(covarianceMatrix)
projectedData = projectData(normalizedData, u, maxK = 100)
recoveredData = recoverData(projectedData, u, maxK = 100)
recoveredData = deNormalize(recoveredData, mean, variance)
sliderEye = projectedData[0]
sliderHandler = SliderHandler(sliderEye, mean, variance, (30,40))

while True:
    cv2.waitKey(10)