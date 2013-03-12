import numpy as np
from numpy.linalg import *
from matplotlib import pylab
import math
import os
import pickle
import tutorial

#
# The intention with this file is to have a reference for a python version of the
# Octave tutorial given in the online Stanford ML course.
# By following the Octave tutorial from the Stanford ML course, one should
# recognize the examples demonstrated below.
#
# For matrix operations, scipy.org has a good reference
# for MatLab equalivants in NumPy: http://www.scipy.org/NumPy_for_Matlab_Users
#
# Also check http://www.scipy.org/Numpy_Example_List
#
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
A = np.matrix('1 2; 3 4; 5 6') #Convenient declaration
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
M = 2 * np.ones((2,4)) #fill with ones and multiply by two

# Random numbers row by column
M = np.random.rand(2,3)
# Gaussian, mean=0 and standard deviation=1
M = np.random.randn(1,3)

M = -6 * math.sqrt(10) * np.random.randn(1,500)
# Values binned to two decimals, otherwise two values are very unlikely to be the same
# Notice that the matrix is put into a single list for display with pylab.hist
M = [math.ceil(col*100)/100 for row in M for col in row]
# pylab.hist(M, bins=50, range=None, normed=1)
# pylab.show()

#3 by 3 or 6 by 6 identity matrix
M = np.identity(3)
M = np.identity(6)

# Matrix dimension tuple
np.shape(np.ones((2,3)))

#current working directory
os.getcwd()

tutorial.createDataFiles()

#Read file, strip new line character
f = open('featuresX.dat', 'r')
features = map(lambda x: x.strip(), f.readlines())
print 'featuresX: ', features
f.close()
f = open('priceY.dat', 'r')
features = map(lambda x: x.strip(), f.readlines())
print 'priceY: ', features

# First 10 elements
v = features[0:10]
print 'first 10 elements:', v

#Create hello.mat, write first 10 elements to it, and read it in again
f2 = open('hello.mat', 'w')
pickle.dump(v, f2, protocol=pickle.HIGHEST_PROTOCOL) #efficient unreadable format protocol
f2.close()
f2 = open('hello.mat', 'r')
v1 = pickle.load(f2)
f2.close()
print 'Unpickled: ', v1

# Create hello.mat in readable format
f2 = open('hello_readable.mat', 'w')
for i in v:
  f2.write("%s\n" % i)
f2.close()

# Index, like 3,2 in Octave tutorial, but here array is zero indexed
M = np.matrix("[1 2; 3 4; 5 6]")
print 'M row 3 col 2 \n', M[2,1] #prints 6
print 'M row 2 \n', M[1,:]
print 'M col 2 \n', M[:,1]
print 'M first and last row, all col \n', M[(0,2),:]

#Assignment
col = [[10],[11],[12]]
M[:,1] = col
print 'Assigned column: \n', M
print 'Append column \n', np.append(M, col, axis=1)
print 'Append row \n', np.append(M, [[14,15]], axis=0)

# Matrix to vector
A = M.reshape(-1)
print 'Row vector: \n', A
print 'Column: \n', A.getT() #get the transpose

# Append two matrices
A = np.matrix("1 2; 3 4; 5 6")
B = np.matrix("5 6; 7 8; 9 10")
C = np.append(A, B, axis=1) #on the side
D = np.append(A, B, axis=0) #on top
print 'Append two matrices on the side: \n', C
print 'Append two matrices on the top: \n', D
# print np.log(A)
# print np.exp(A)
# val = np.max(np.exp(A))
# index = np.argmax(np.exp(A))
# print val
# print index
# print np.abs(A)
# print np.sum(A)
# print np.floor(A)
# print A.max(axis=0)
# print np.identity(9)
# print np.flipud(np.identity(9))
# print pinv(A)
# print np.abs(np.round(pinv(A) * A))

#Plotting
# t = np.arange(1,120,0.01)
# y1 = np.sin((np.arange(1,120,0.01)))
# print y1
# pylab.plot(t,y1)
# pylab.show()
# A = np.matrix("[1 2]")
# X = np.matrix("[5 5]")
# print 'Vectorised sum: ', np.inner(A, X)