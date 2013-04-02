__author__ = 'ahkj'

from pca import *
from svm import *
import matplotlib.pyplot as plt
import time

#pct of data used.
dataSize = 0.5
#pct of data used for training, and consequently testing
defaultTrainingSize = 0.8
#Dimensionality reduction we use from the PCA
reducedDim = 100


def predict(clf, data, targets):
    '''Returns number of correctly classified and misclassified elements in the data, as predicted by a classifier.
    :param clf: Classifier/Estimator from scikit-learn
    :param data: Test data
    :param targets: Target values
    '''
    correct = 0
    for index, sample in enumerate(data):
        if targets[index] == clf.predict(sample)[0]:
            correct += 1
    return correct, len(data) - correct


def runConsole(data, targets, printWithoutPCA=False):
    '''Run with console output. '''
    #Samples for training
    training = int(len(data) * defaultTrainingSize)
    print 'training ', training
    #Samples for test
    test = (len(data) - training)
    print 'test: ', test

    print 'Running with console output.\nTraining data: %s Test data: %s' % (training, test)
    (trainingData, trainingTargets, testData, testTargets) = getTrainingAndTestSets(training, test, data, targets,
                                                                                    reducedDim)
    if printWithoutPCA:
        print '\nWithout PCA: '
        t0 = time.time()
        clf = classifier(data[:training], targets[:training])
        print 'Time learning: ', time.time() - t0
        t0 = time.time()
        print 'Classified: %s \nMisclassified: %s' % (predict(clf, data[-test:], targets[-test:]))
        print 'Time predicting: ', time.time() - t0

    print '\nWith PCA: '
    t0 = time.time()
    clf = classifier(trainingData, trainingTargets)
    print 'Time learning: ', time.time() - t0
    print 'Classified: %s \nMisclassified: %s' % predict(clf, testData, testTargets)
    print 'Time predicting: ', time.time() - t0


def runErrorVersusDimension(data, targets):
    '''Performs several runs adjusting the dimensionality reduction. Dimension versus test error is plotted.'''
    #Size of test and training data
    trainingSize = int(len(data) * defaultTrainingSize)
    testSize = (len(data) - trainingSize)
    errors = []

    #Normalize and do PCA
    trainingData = data[:int(len(data) * trainingSize)]
    trainingTargets = targets[:int(len(data) * trainingSize)]
    testData = eyeData[-testSize:]
    testTargets = targets[-testSize:]

    trainingNormalizedData, trainingMean, std = featureNormalize(trainingData)
    trainingCovarianceMatrix = getCovarianceMatrix(trainingNormalizedData)
    (u1, s1, v1) = np.linalg.svd(trainingCovarianceMatrix)

    testNormalizedData, testMean, std = featureNormalize(testData)
    testCovarianceMatrix = getCovarianceMatrix(testNormalizedData)
    (u2, s2, v2) = np.linalg.svd(testCovarianceMatrix)

    #We iterate from 1 to max dimension
    dims = [x + 1 for x in range(len(eyeData[0]))]
    dims = dims[:200]
    for x in dims:
        print 'Doing dim: ', x
        trainingProjectedData = projectData(trainingNormalizedData, u1, maxK=x)
        testProjectedData = projectData(testNormalizedData, u2, maxK=x)
        clf = classifier(trainingProjectedData, trainingTargets)
        correct, mis = predict(clf, testProjectedData, testTargets)
        print 'Correct: %s Mis: %s' % (correct, mis)
        errors.append(mis / float(len(testProjectedData)))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(dims, errors)
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('dimension')
    plt.ylabel('pct wrong classification')
    plt.show()


def runErrorVersusTrainingSize(data, targets):
    '''Performs several runs adjusting the amount of training data used and
    plots the error rate on the test data, which stays as a fixed amount.
    Error percentage is percentage of of wrong classifications over test data size
    '''

    pct = 0.9
    decrement = 0.005
    errors = []
    inSample = []

    #Size of test and training data
    trainingSize = int(len(data) * pct)
    testSize = (len(data) - trainingSize)

    #Normalize and do PCA
    (trainingData, trainingTargets, testData, testTargets) = getTrainingAndTestSets(trainingSize,
                                                                                    testSize,
                                                                                    data,
                                                                                    targets,
                                                                                    reducedDim)

    #TODO, 800 is arbitrary, I get a crash when training size gets close to 0
    while 0 < pct and 800 < trainingSize:
        clf = classifier(trainingData[:trainingSize], trainingTargets[:trainingSize])
        correct, mis = predict(clf, testData, testTargets)
        # print 'Correct: %s Mis: %s' % (correct, mis)
        # print 'Training size %s Test size: %s' % (training, test)
        errors.append(mis / float(len(testData)))
        inSample.append(len(trainingData[:trainingSize]))
        pct -= decrement
        trainingSize = int(len(data) * pct)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(inSample, errors)
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Training sample size')
    plt.ylabel('Error pct')
    plt.show()

def getTrainingAndTestSets(trainingSize, testSize, data, targets, dim):
    '''Returns training data, training targets, testData, testTargets
       Training and test data will be run through PCA separately.
    '''
    #Do PCA on test and training data separately
    trainingData = data[:int(len(data) * trainingSize)]
    trainingTargets = targets[:int(len(data) * trainingSize)]
    testData = data[-testSize:]
    testTargets = targets[-testSize:]

    trainingNormalizedData, trainingMean, trainingVariance = featureNormalize(trainingData)
    trainingCovarianceMatrix = getCovarianceMatrix(trainingNormalizedData)
    (u1, s1, v1) = np.linalg.svd(trainingCovarianceMatrix)
    trainingProjectedData = projectData(trainingNormalizedData, u1, maxK=dim)

    testNormalizedData, testMean, testVariance = featureNormalize(testData)
    testCovarianceMatrix = getCovarianceMatrix(testNormalizedData)
    (u2, s2, v2) = np.linalg.svd(testCovarianceMatrix)
    testProjectedData = projectData(testNormalizedData, u2, maxK=dim)
    return trainingProjectedData, trainingTargets, testProjectedData, testTargets

def plotClasses(data, targets):
    #Find one of each class
    i = 0
    d = {}
    (trainingData, trainingTargets, testData, testTargets) = getTrainingAndTestSets(len(data),
                                                                                    100,
                                                                                    data,
                                                                                    targets,
                                                                                    2)
    sym = [' ', 'ro', 'g>', 'b<', 'c.']
    for index, trainingData in enumerate(eyeData):
        cls = trainingTargets[index]
        plt.plot(trainingData[0], trainingData[1], sym[cls])
    plt.show()

def testRun(eyeData, targets):
    # Will separate training and test data itself
    # runErrorVersusTrainingSize(eyeData, targets)
    plotClasses(eyeData, targets)
    # runConsole(eyeData, targets)
    # runErrorVersusDimension(eyeData, targets)

#Load eye data, and reduced amount of data if specified
loader = EyeVideoLoader()
(eyeData, targets) = loader.loadDataFromVideos()
print 'Eye data: ', len(eyeData)
eyeData = eyeData[:int(len(eyeData) * dataSize)]
print 'Reduced eye data: ', len(eyeData)
targets = targets[:int(len(targets) * dataSize)]
testRun(eyeData, targets)