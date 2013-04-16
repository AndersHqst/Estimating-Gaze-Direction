import scipy.cluster.vq as vq
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def normalizeImage(image):
    pupil = findPupil(image)

    if pupil is None:
        return None
    
    eyeCorners = findEyeCorners(image, pupil)
    #cv2.circle(image, tuple(eyeCorners[0]), 8, (0,0,0), 10)
    #cv2.circle(image, tuple(eyeCorners[1]), 8, (0,0,0), 10)

    image = cropImage(image, pupil, eyeCorners, 360, 240)

    cv2.imshow("Debug", image)
    cv2.imwrite("cropped.jpg", image)
    cv2.waitKey(0)

    return image


def cropImage(image, pupil, eyeCorners, width, height):
    centreY = pupil[1]
    imageHeight = image.shape[0]
    top = centreY - height/2
    top = max((top, 0))
    top = min((top, imageHeight - height))

    centreX = np.average(eyeCorners[:,0])
    imageWidth = image.shape[1]
    left = centreX - width/2
    left = max((left, 0))
    left = min((left, imageWidth - width))

    image = image[top:top+height, left:left+width]
    return image    


def findEyeCorners(image, pupil):
    pupilX = pupil[0]
    leftImage = image[:, 0:pupilX]
    rightImage = image[:, pupilX:]
    leftTemplate = cv2.imread("leftTemplate.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    rightTemplate = cv2.imread("rightTemplate.png", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    leftCorner = findTemplate(leftImage, leftTemplate, pupil, "Left")
    rightCorner = findTemplate(rightImage, rightTemplate, pupil, "Right")

    rightCorner[0] += pupilX

    return np.array([leftCorner, rightCorner])


def findTemplate(pattern, template, pupil, windowName = None):
    
    idealPosition = np.array([pupil[1] - template.shape[0]/float(2), # row: pupil Y minus half the template
                              (pattern.shape[1] - template.shape[1]) / float(2)]) / float(2) # downscale

    scaledPattern = cv2.pyrDown(pattern)
    scaledTemplate = cv2.pyrDown(template)
    
    # match, x, y
    ideals = np.array([1, idealPosition[1], idealPosition[0]]).reshape(3, 1, 1)
    weights = np.array([1, 1, 1]).reshape(3, 1, 1)

    match = cv2.matchTemplate(scaledPattern, scaledTemplate, cv2.TM_CCOEFF_NORMED) # cross correlation
    values = np.zeros([3, match.shape[0], match.shape[1]])
    
    values[0] = match
    indices = np.indices(match.shape)
    values[1] = indices[1]
    values[2] = indices[0]

    errors = (values - ideals) * weights / ideals
    errors = np.sqrt(errors[0]**2 + errors[1]**2 + errors[2]**2)

    #if windowName is not None:
    errorDisplay = errors / np.max(errors)
    errorDisplay = cv2.pyrUp(errorDisplay)
    cv2.imshow("Debug", errorDisplay)
    cv2.waitKey(0)

    y, x = np.unravel_index(np.argmin(errors), errors.shape)
    
    x = x * 2
    y = y * 2

    x += template.shape[1] / 2
    y += template.shape[0] / 2

    return [x, y]


def findPupil(image):
    #threshold = getPupilThresholdWithClustering(image, K = 20)
    threshold = 62
    value, binaryImage = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    binaryImage = applyMorphology(binaryImage)

    cv2.imshow("Debug", binaryImage)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(np.copy(binaryImage), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pupil = getPupilCentre(contours, image.shape)
    return pupil


def getPupilCentre(contours, imageSize):
    pupils = []

    # area, extent, x, y
    ideals = np.array([3000.0, 0.7, imageSize[0]/2.0, imageSize[0]/2.0])
    weights = [1, 2, 1, 1]
    maxError = 1

    for contour in contours:
        contour = contour.astype('int')
        area = cv2.contourArea(contour)
        extent = getExtent(contour)
        if (extent > 0.5 and area >= 100 and area <= 10000):
            centroid = getCentroid(contour)
            if (centroid[0] > 20 and centroid[1] > 20 and centroid[0] < (imageSize[1] - 20) and (centroid[1] < imageSize[0])):
                values = np.array([area, extent, centroid[0], centroid[1]])
                errors = (values - ideals) * weights / ideals
                error = np.linalg.norm(errors)
                pupils.append((error, centroid))

    imageCentre = np.array(imageSize) / 2
    pupils = sorted(pupils, key = lambda t: t[0])

    if (len(pupils) == 0 or pupils[0][0] > maxError):
        return None

    return pupils[0][1]


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
    return [int(m['m10']/m['m00']), int(m['m01']/m['m00'])]



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

    plotClusters(centroids, features, M, N)

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
