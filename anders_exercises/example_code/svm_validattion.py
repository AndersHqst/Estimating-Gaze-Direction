import numpy as np
import svm

eyeData = np.load('eyeData.npy')
targets = np.load('targets.npy')
people = np.load('people.npy')

def classifier(data, targets, C, gamma, kernel):
    '''Return a SVM that that has learned from the training data provided.
    TODO: Understand SVM parameters, and optimize with eg cross-validation
    :param data:
    :param targets:
    '''
    clf = svm.SVC(C = C, gamma = gamma, kernel = kernel)
    #clf = svm.NuSVC(nu=0.2, gamma = gamma, kernel = kernel)

    #clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    #              gamma=0.0001, kernel='rbf', max_iter=-1, probability=False,
    #              shrinking=True, tol=0.001, verbose=False)
    clf.fit(data, targets)
    return clf

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

    # Dict of of misclassified images
    miss = []
    for index, result in enumerate(testResults):
        if testTargets[index] != result:
            miss.append({ 'eyeData': testData[index], 'target': testTargets[index], 'classification': result, 'testPerson': testPerson})

    #correct = np.sum((testTargets-1)/2 == (testResults-1)/2) / float(len(testResults))
    correct = np.sum(testTargets == testResults) / float(len(testResults))

    #print classification_report(testTargets, testResults
    return classifier, correct, miss


def create_figures(misses):
    print 'Create figure. Misses: ', len(misses[0])
    targetCount = {1:0, 2:0, 3:0, 4:0}
    miss_classification = {1:0, 2:0, 3:0, 4:0}
    # Count classification error by classification, against what they should have been
    realtive_errors = {
        1: {1:0, 2:0, 3:0, 4:0},
        2: {1:0, 2:0, 3:0, 4:0},
        3: {1:0, 2:0, 3:0, 4:0},
        4: {1:0, 2:0, 3:0, 4:0}
        }
    for m in misses:
        for sample in m:
            targetCount[sample['target']] += 1
            classCount[sample['classification']] += 1
            relative_errors[sample['classification']][sample['target']] += 1

    #Save images
    #
    for eye in misses[0]:
        img_eye = eye.reshape((28,42))


# img_eye = eye.reshape((28,42))
    tex_figures = open('miss_classifications')

def build_table(miss_classification, realtive_errors):
    """Build a one colum latex table.

    :param classifications: dictionary of real classification against misclassification
    """
    table = "\\hhtab{p{130pt}p{50pt}}\n"
    table += "{\n"
    table += "\\toprule\n"
    table += "Miss classification & Count & Target 1 & Target 2 & Target 3 & Target 4\\\\\n"
    table += "\\midrule\n"

    for key in miss_classification.keys():
        count = miss_classification[key]
        #Percentages distributed over their real targets
        t1 = 100 * (realtive_errors[key][1] / float(count))
        t2 = 100 * realtive_errors[key][2] / float(count))
        t3 = 100 * (realtive_errors[key][3] / float(count))
        t4 = 100 * (realtive_errors[key][4] / float(count))
        table += '%s & %s\% & %s\% & %s\% & %s\%' % (count, t1, t2, t3, t4) + "\\\\\n"
    table += "\\bottomrule\n"
    table += "}{" + caption + "}{tab:" + title + "}"
    print "table string: ", table


def build_figure(clazz, miss):
    name = "class_" + clazz + "_miss_" + miss
    fig = """\begin{figure}[h!]""" + "\n"
    fig += """\begin{center}""" + "\n"
    fig += """\includegraphics[width=0.60\columnwidth]{figures/""" + name + ".png}\n"
    fig += """\end{center}""" + "\n"
    fig += """\caption{ Class: """ + clazz + " misclassification: " + miss + "}\n"
    fig += """\label{fig:""" + name + "}\n"
    fig += """\end{figure}"""
    return fig, name + ".png"

#Best parameters
# k=131, C=1000,0 ; gamma=0,001 ; rbf -> 0,857739014385



# eye = misses[0][0]
# img_eye = eye.reshape((28,42))

# imshow(img_eye, cmap='gray')

# #For plotting support vectors
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# for vec in clf.support_vectors_:
#     val = clf.predict(vec)
#     col = ''
#     if val == 1:
#         col = 'bo'
#     elif val == 2:
#         col = 'ro'
#     elif val == 3:
#         col = 'go'
#     elif val == 4:
#         col = 'yo'
#     ax1.plot(vec[0], vec[1], col)