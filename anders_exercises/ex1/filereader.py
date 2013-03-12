
pop=[]
profit=[]
def readFile():
    f=open('ex1data1.txt', 'r')
    #added dummy variable for theta0
    for x in enumerate(f.readlines()):
        pop.append([1, float(x[1].partition(',')[0])])
        profit.append(float(x[1].partition(',')[2]))
    return (pop,profit)
