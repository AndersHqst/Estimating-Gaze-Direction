import numpy as np
from matplotlib import pylab
import math

#
# The intention with this file is to have a reference for a python version of the
# Octave tutorial given in the online Stanford ML course.
# By following the Octave tutorial from the Stanford ML course, one should
# recognize the operations demonstrated below in the corresponding order.
#
# All examples are put here.
#

#Arithmetic operators
print 5 + 6
print 3 - 2
print 5 * 8
print 1 / 2.0
print 2 ** 6

#Logical operators
print 1 == 2
print 1 != 2
print True and True
print False or True

#Variables
a = 3
c = (3>=1)
print c
a = math.pi
print '2 decimals %.2f' % a
print '2 decimals %.6f' % a

# Matrices
A = np.matrix([[1,2], [3,4], [5,6]]) #3 by 2 matrix
A = np.matrix('1 2; 3 4; 5 6') #Alternative declaration
v = np.matrix([1,2,3]) #Row vector
v = np.matrix([[1],[2],[3]]) #column vector

M = np.matrix([[1,2,3,4,5],[6,7,8,9,10]])
M.fill(1) #Fill matrix with 1s

# A row vector with 11 numbers evenly spaced between 1 and 2
# [ 1.   1.1  1.2  1.3  1.4  1.5  1.6  1.7  1.8  1.9  2. ]
A = np.linspace(1, 2, 11, endpoint=True, retstep=False)

# Numbers generated in a half open interval (last number is excluded)
# [0 1 2 3 4 5 6 7 8 9]
A = np.arange(0, 10, 1)

# Row by column matrix prefilled, defaults to floats
M = np.ones((2,4))
M = np.zeros((1,2), dtype=np.int8) #require type
M = 2 * np.ones((2,4)) #fill with twos

# Random numbers row by column
M = np.random.rand(2,3)
# Gaussian, mean=0 and standard deviation=1
M = np.random.randn(1,3)

M = -6 * math.sqrt(10) * np.random.randn(1,1000)
# Values binned to two decimals, otherwise two values are very unlikely to be the same
M = [math.ceil(x*100)/100 for l in M for x in l]
pylab.hist(M, bins=50)
# pylab.show()

#3 by 3 or 6 by 6 identity matrix
M = np.identity(3)
M = np.identity(6)



