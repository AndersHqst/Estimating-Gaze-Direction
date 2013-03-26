__author__ = 'ahkj'

from pca import *
from svm import *
import matplotlib.pyplot as plt

#pct of data used.
dataSize = 0.4
#pct of data used for training, and consequently testing
trainingSize = 0.95
#Dimensionality reduction we use from the PCA
reducedDim = 10

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


def runConsole(eyeData, targets):
    '''Run with console output. '''
    #Training samples to be used
    training = int(len(eyeData) * trainingSize)
    print 'training ', training
    #Test data. Minus one is to avoid sharing the middle element
    test = (len(eyeData) - training) - 1
    print 'test: ', test

    print 'Running with console output.\nTraining data: %s Test data: %s' % (training, test)
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

def runErrorDimension(eyeData, targets):
    '''Performs several runs adjusting the dimensionality reduction. Dimension versus test error is plotted.'''
    misclassified = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(misclassified, [x for x in range(len(eyeData))])
    ax.grid(True)
    plt.show()

def testRun():
    loader = EyeVideoLoader()
    (eyeData, targets) = loader.loadDataFromVideos()
    print 'Eye data: ', len(eyeData)
    eyeData = eyeData[:int(len(eyeData) * dataSize)]
    print 'Reduced eye data: ', len(eyeData)
    targets = targets[:int(len(targets) * dataSize)]
    runConsole(eyeData, targets)

testRun()