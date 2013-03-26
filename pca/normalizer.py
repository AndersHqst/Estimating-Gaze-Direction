import scipy.cluster.vq as vq
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def normalizeImage(image):
    # threshold = getPupilThresholdWithClustering(image, K = 20)
    threshold = 62
    value, binaryImage = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    binaryImage = applyMorphology(binaryImage)
    
    contours, hierarchy = cv2.findContours(np.copy(binaryImage), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    pupil = getPupilCentre(contours, image.shape)

    if pupil is None:
        return None
    else:
        image = cropImage(image, pupil, (160,120))
        cv2.imshow("Threshold", image)
        cv2.waitKey(10)
        return image

        # cv2.circle(binaryImage, pupil, 8, (0,0,0), 10)

    # cv2.imshow("Threshold", binaryImage)
    # cv2.waitKey(10)


def cropImage(image, centre, radi):
    left = max(0, centre[0] - radi[0])
    top = max(0, centre[1] - radi[1])
    return image[top:top+radi[1]*2, left:left+radi[0]*2]



def getPupilCentre(contours, imageSize):
    pupils = []

    for contour in contours:
        contour = contour.astype('int')
        area = cv2.contourArea(contour)
        if (getExtent(contour) > 0.5 and area >= 100 and area <= 10000):
            pupils.append(getCentroid(contour))

    imageCentre = np.array(imageSize) / 2
    pupils = sorted(pupils, key = lambda p: np.linalg.norm(p - imageCentre))

    if (len(pupils) == 0):
        return None

    return pupils[0]


def applyMorphology(binaryImage):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, kernel)
    return binaryImage

def getExtent(contour):
    area = cv2.contourArea(contour)
    bounds = cv2.boundingRect(contour)
    return area / (bounds[2]*bounds[3])

def getCentroid(contour):
    m = cv2.moments(contour)
    return (int(m['m10']/m['m00']), int(m['m01']/m['m00']))


def getPupilThresholdWithClustering(gray,K=2, distanceWeight=2, resizeTo=(40,40)):
    ''' Detects the pupil in the image, gray, using k-means
        gray            : gray scale image
        K               : Number of clusters
        distanceWeight  : Defines the weight of the position parameters
        reSize          : the size of the image to do k-means on
    '''
    
    smallI = cv2.resize(gray, resizeTo)

    M,N = smallI.shape
    #Generate coordinates in a matrix
    X,Y = np.meshgrid(range(M),range(N))

    #Make coordinates and intensity into one vectors
    z = smallI.flatten()
    x = X.flatten()
    y = Y.flatten()

    # make a feature vectors containing (x,y,intensity)
    features = np.zeros((len(x),3))
    features[:,0] = z;
    features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less than intensity
    features[:,2] = x/distanceWeight;
    features = np.array(features,'f')

    # cluster data
    centroids,variance = vq.kmeans(features,K)

    #plotClusters(centroids, features, M, N)

    centroidsByPupilCandidacy = sorted(centroids, key = lambda c: evaluateCentroid(c, resizeTo))
    
    return centroidsByPupilCandidacy[-1][0] + 10
    

def evaluateCentroid(centroid, shape):
    darkness = 255 - centroid[0]
    distance = np.linalg.norm(centroid[1:3] - np.array(shape)/2)
    return darkness - distance**1.6


def plotClusters(centroids, features, M, N):
    label,distance = vq.vq(features,centroids)
    
    # re-create image
    labelIm = np.array(np.reshape(label,(M,N)))
    
    f = plt.figure(1)
    plt.imshow(labelIm)
    f.canvas.draw()
    f.show()
