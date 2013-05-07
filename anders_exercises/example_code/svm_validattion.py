import numpy as np
import svm

eyeData = np.load('eyeData.npy')
targets = np.load('targets.npy')
people = np.load('people.npy')

def crossValidate(eyeData, people, targets, k = 2, C = 1, gamma = 1e-8, kernel = 'rbf'):
    results = []
    misses = []
    clfs = []
    start = time.time()
    for testPerson in range(np.max(people)):
        classifier, correctFraction, miss = validate(eyeData, people, targets, testPerson, k, C, gamma, kernel)
        results.append(correctFraction)
        clfs.append(classifier)
        misses.append(miss)
    print 'Validation time: ', time.time() - start

    return clfs, results, misses

def validate(eyeData, people, targets, testPerson, k, C, gamma, kernel):
    trainingIndices = np.nonzero(people != testPerson)
    testIndices = np.nonzero(people == testPerson)

    trainingData = eyeData[trainingIndices]
    trainingTargets = targets[trainingIndices]
    testData = eyeData[testIndices]
    testTargets = targets[testIndices]

    # normalize training data, get mean
    (normalizedTraining, mean, variance) = featureNormalize(trainingData, doScale = False)

    # normalize test data with mean from above
    normalizedTest = testData - mean

    # run PCA with some value of k to get (u,s,v) from training data
    covarianceMatrix = getCovarianceMatrix(normalizedTraining)
    (u, s, v) = np.linalg.svd(covarianceMatrix)

    # project training data & test data
    projectedTraining = projectData(normalizedTraining, u, k)
    projectedTest = projectData(normalizedTest, u, k)

    # learn through projected training data
    classifier = svm.classifier(projectedTraining, trainingTargets, C, gamma, kernel)

    # try to predict projected test data
    testResults = classifier.predict(projectedTest)

    miss = []
    for index, result in enumerate(testResults):
        if testTargets[index] != result:
            miss.append({ 'eyeData': testData[index], 'target': testTargets[index], 'classification': result, 'testPerson': testPerson})

    #correct = np.sum((testTargets-1)/2 == (testResults-1)/2) / float(len(testResults))
    correct = np.sum(testTargets == testResults) / float(len(testResults))

    #print classification_report(testTargets, testResults)

    return classifier, correct, miss

def classifier(data, targets, C, gamma, kernel):
    '''Return a SVM that that has learned from the training data provided.
    TODO: Understand SVM parameters, and optimize with eg cross-validation
    :param data:
    :param targets:
    '''
    # clf = svm.SVC(C = C, gamma = gamma, kernel = kernel)
    clf = svm.NuSVC(nu=0.2, gamma = gamma, kernel = kernel)

    #clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    #              gamma=0.0001, kernel='rbf', max_iter=-1, probability=False,
    #              shrinking=True, tol=0.001, verbose=False)
    clf.fit(data, targets)
    return clf


targetCount = {1:0, 2:0, 3:0, 4:0}
classCount = {1:0, 2:0, 3:0, 4:0}
for m in misses:
    for sample in m:
        targetCount[sample['target']] += 1
        classCount[sample['classification']] += 1

eye = misses[0][0]
img_eye = eye.reshape((28,42))

imshow(img_eye, cmap='gray')

#For plotting support vectors
fig = plt.figure()
ax1 = fig.add_subplot(111)
for vec in clf.support_vectors_:
    val = clf.predict(vec)
    col = ''
    if val == 1:
        col = 'bo'
    elif val == 2:
        col = 'ro'
    elif val == 3:
        col = 'go'
    elif val == 4:
        col = 'yo'
    ax1.plot(vec[0], vec[1], col)