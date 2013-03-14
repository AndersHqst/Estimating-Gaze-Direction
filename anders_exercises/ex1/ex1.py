import numpy as np
from numpy.linalg import *
from matplotlib import pylab
from math import *
# import unittest
import filereader
from plotting import *
import matplotlib.pyplot as plt

#learning rate
a=0.03

def initialTheta():
    return [0.1,-2]

def hyp(theta,x):
    return sum([i*j for i,j in zip(theta, x)])

def cost(theta, data, target):
    return 1.0/(2*len(data))*sum([(hyp(theta, x)-y)**2 for x,y in zip(data, target)])

def thetaGradient(theta, data, target, partial):
    return a*(1.0/(len(data)))*sum([(hyp(theta, x)-y)*x[partial] for x,y in zip(data, target)])

def iterativeBatchGradientDescent(theta, data, target, a):
    for index,t in enumerate(theta):
        theta[index]=t-a*thetaGradient(theta, data, target, index)
    return theta

def run():
    pop, profit = filereader.readFile();
    plotdata(pop, profit)
    theta=initialTheta()
    c=cost(theta, pop, profit)
    print 'Initial theta0: ', theta[0]
    print 'Initial theta1: ', theta[1]
    print 'Initial cost: ', c
    print 'Learning rate: ', a
    for i in range(1000):
        theta=iterativeBatchGradientDescent(theta, pop, profit, a)
        # plotCost(c, i)
        # c=cost(theta, pop, profit)
        if i % 100 == 0:
            plotLine(theta[1], theta[0])
            print 'Cost iteration: ', (cost(theta, pop, profit), i)
    plotLine(theta[1], theta[0])
    print 'Cost: ', cost(theta, pop, profit)
    print 'Final theta0: ', theta[0]
    print 'Final theta1: ', theta[1]
    plt.show()

run()



# class TestSequenceFunctions(unittest.TestCase):
#     def testhyp(self):
#         self.assertEqual(hyp([1,2,3],[1,2,3]), 14)

#     def testhyp2(self):
#         self.assertEqual(hyp([2,3,4],[1,2,3]), 20)

#     def testcost(self):
#         self.assertEqual(cost([1,1,1], [[1],[1],[1]], [1,1,1]), 0)

# unittest.main()