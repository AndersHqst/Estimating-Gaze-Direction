__author__ = 'ahkj'

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def classifier(data, targets, C, gamma, kernel):
    '''Return a SVM that that has learned from the training data provided.
    TODO: Understand SVM parameters, and optimize with eg cross-validation
    :param data:
    :param targets:
    '''
    clf = svm.SVC(C = C, gamma = gamma, kernel = kernel)

    #clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    #              gamma=0.0001, kernel='rbf', max_iter=-1, probability=False,
    #              shrinking=True, tol=0.001, verbose=False)
    clf.fit(data, targets)
    return clf




def runGridSearch(projectedTraining, trainingTargets):
    classifier = svm.SVC()
    parameterGrid = [
        #{'C': [1, 10, 100, 1000], 'kernel': ['rbf', 'linear'], 'gamma': [0.01, 0.001, 0.0001, 0.00001]}
        {'C': [1, 100], 'kernel': ['rbf', 'linear'], 'gamma': [0.0001]}
    ]
    search = GridSearchCV(classifier, parameterGrid, score_func = precision_score)
    search.fit(projectedTraining, trainingTargets)

    print search
    raw_input()

