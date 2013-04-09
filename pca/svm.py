__author__ = 'ahkj'

from sklearn import svm

def classifier(data, targets):
    '''Return a SVM that that has learned from the training data provided.
    TODO: Understand SVM parameters, and optimize with eg cross-validation
    :param data:
    :param targets:
    '''
    clf = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  gamma=0.0001, kernel='rbf', max_iter=-1, probability=False,
                  shrinking=True, tol=0.001, verbose=False)
    clf.fit(data, targets)
    return clf