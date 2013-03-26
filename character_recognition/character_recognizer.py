__author__ = 'ahkj'

from sklearn import datasets
from sklearn import svm
import numpy as np

test = 100
mis = 0

digits = datasets.load_digits()

print 'type: ', type(digits.data[0])
print 'type: ', type(digits.target)

print 'data: ', digits.data[0]
print 'shape: ', np.shape(digits.data[0])
print 'target: ', digits.target

clf = svm.SVC(gamma=0.0001, C=100.)
print clf.fit(digits.data[:-test], digits.target[:-test])

for i in range(test):
    # print 'result: %s actual: %s' %(clf.predict(digits.data[i *- 1]), digits.target[i *- 1])
    if clf.predict(digits.data[i * -1])[0] != digits.target[i * -1]:
        mis += 1

print 'digits: ', len(digits.data[:-test])
print 'missed: ', mis