__author__ = 'ahkj'

from pca import *
from svm import *
import matplotlib.pyplot as plt
import time

#pct of data used.
dataSize = 1
#pct of data used for training, and consequently testing
trainingSize = 0.90
#Dimensionality reduction we use from the PCA
reducedDim = 20


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


def runConsole(eyeData, targets, printWithoutPCA=False):
    '''Run with console output. '''
    #Samples for training
    training = int(len(eyeData) * trainingSize)
    print 'training ', training
    #Samples for test
    test = (len(eyeData) - training)
    print 'test: ', test

    print 'Running with console output.\nTraining data: %s Test data: %s' % (training, test)

    if printWithoutPCA:
        print '\nWithout PCA: '
        t0 = time.time()
        clf = classifier(eyeData[:training], targets[:training])
        print 'Time learning: ', time.time() - t0
        t0 = time.time()
        print 'Classified: %s \nMisclassified: %s' % (predict(clf, eyeData[-test:], targets[-test:]))
        print 'Time predicting: ', time.time() - t0

    normalizedData, mean, variance = featureNormalize(eyeData)
    covarianceMatrix = getCovarianceMatrix(normalizedData)
    (u, s, v) = np.linalg.svd(covarianceMatrix)
    projectedData = projectData(normalizedData, u, maxK=reducedDim)

    print '\nWith PCA: '
    t0 = time.time()
    clf = classifier(projectedData[:training], targets[:training])
    print 'Time learning: ', time.time() - t0
    print 'Classified: %s \nMisclassified: %s' % predict(clf, projectedData[-test:], targets[-test:])
    print 'Time predicting: ', time.time() - t0


def runErrorVersusDimension(eyeData, targets):
    '''Performs several runs adjusting the dimensionality reduction. Dimension versus test error is plotted.'''
    #Size of test and training data
    training = int(len(eyeData) * trainingSize)
    test = (len(eyeData) - training)
    errors = []

    #Normalize and do PCA
    normalizedData, mean, variance = featureNormalize(eyeData)
    covarianceMatrix = getCovarianceMatrix(normalizedData)
    (u, s, v) = np.linalg.svd(covarianceMatrix)

    #We iterate from 1 to max dimension
    dims = [x + 1 for x in range(len(eyeData[0]))]
    dims = dims[:50]
    for x in dims:
        print 'Doing dim: ', x
        projectedData = projectData(normalizedData, u, maxK=x)
        clf = classifier(projectedData[:training], targets[:training])
        correct, mis = predict(clf, projectedData[-test:], targets[-test:])
        print 'Correct: %s Mis: %s' % (correct, mis)
        errors.append(mis / float(len(projectedData[-test:])))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(dims, errors)
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('dimension')
    plt.ylabel('pct wrong classification')
    plt.show()


def runErrorVersusTrainingSize(eyeData, targets):
    '''Performs several runs adjusting the amount of training data used and
    plots the error rate on the test data, which stays as a fixed amount.
    Error percentage is percentage of of wrong classifications over test data size
    '''

    pct = 0.9
    decrement = 0.005
    errors = []
    inSample = []

    #Size of test and training data
    training = int(len(eyeData) * pct)
    test = (len(eyeData) - training)

    #Normalize and do PCA
    normalizedData, mean, variance = featureNormalize(eyeData)
    covarianceMatrix = getCovarianceMatrix(normalizedData)
    (u, s, v) = np.linalg.svd(covarianceMatrix)
    projectedData = projectData(normalizedData, u, maxK=reducedDim)

    #TODO, 800 is arbitrary, I get a crash when training size gets close to 0
    while 0 < pct and 800 < training:
        clf = classifier(projectedData[:training], targets[:training])
        correct, mis = predict(clf, projectedData[-test:], targets[-test:])
        # print 'Correct: %s Mis: %s' % (correct, mis)
        errors.append(mis / float(len(projectedData[-test:])))
        inSample.append(len(projectedData[:training]))
        pct -= decrement
        training = int(len(eyeData) * pct)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(inSample, errors)
    ax.grid(True)
    fig.autofmt_xdate()
    plt.xlabel('Training sample size')
    plt.ylabel('Error pct')
    plt.show()


def testRun(eyeData, targets):
    # runConsole(eyeData, targets)
    # runErrorVersusDimension(eyeData, targets)
    runErrorVersusTrainingSize(eyeData, targets)


loader = EyeVideoLoader()
(eyeData, targets) = loader.loadDataFromVideos()
print 'Eye data: ', len(eyeData)
eyeData = eyeData[:int(len(eyeData) * dataSize)]
print 'Reduced eye data: ', len(eyeData)
targets = targets[:int(len(targets) * dataSize)]
testRun(eyeData, targets)