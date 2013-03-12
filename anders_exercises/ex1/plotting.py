import numpy as np
import matplotlib.pyplot as plt

def plotLine(x, b):
    #generate line points
    a=[]
    xs=range(20)
    for i in xs:
        a.append([i,x*i+b])
    plt.plot(*zip(*a))

def plotCost(cost, t):
    plt.scatter(t,cost,s=10, marker='^', c='r')

def plotdata(d1,d2):
    d=[]
    for data in d1:
        d.append(data[1])
    color = np.random.rand(len(d),3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d, d2, c=color, s=80, alpha=0.75)
    ax.set_xlabel('Population', fontsize=20)
    ax.set_ylabel('Profit', fontsize=20)
    ax.set_title('Population to profit')
    ax.grid(True)
    # plt.show()