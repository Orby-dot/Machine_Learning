#Linear regression model

#Purpose: To predict outcome of data under the assumption:
#   1: The observation is linearly correlated to the desired paramter
#   2: There is some noise W ~ N(0,σ^2)
#   3: The model will be of the form: Y = a + bX where Y is the desired paramter and X is the observation

#Goal: To find a and b in the above equation

#Extensions: To allow any polynomial or have multiple variables correlated to the desired parameter

#Background info:
#   Refer to directory called "Background Info"

##WARNING
#   CORRELATION IS NOT CAUSATION. PLEASE KEEP THIS IN MIND WHEN YOU ARE USING THIS MODEL.
#Packages: Limited
from numpy import linalg
import numpy as np
import argparse

#testing scripts for debugging
testScript = True
try:
    import scripts.lineCreation as line
except ImportError:
    testScript = False

#Debug variables
DEBUG = 1
N = 20
TESTS = 10
STD_DEV = 2
MEANX = 0
VARX = 10
PORTION = .5

X = [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10]
Y = [4,7,10,13,16,19,22,25,28,31,34]
Z = [2,3, 4, 5, 6, 7, 8, 9,10,11,12]

def turnToSets(data):
    x=[]
    y=[]
    for i in data:
        x.append(i[0])
        y.append(i[1])

    return [x,y]

def testBasicSets(set,portion):
    accuracy = 0

    trainingSet = []
    trainingSet.append(set[0][0:int(len(set[0])*portion)])
    trainingSet.append(set[1][0:int(len(set[1])*portion)])

    testingSet = []
    testingSet.append(set[0][int(len(set[0])*portion):])
    testingSet.append(set[1][int(len(set[1])*portion):])

    params = linReg.basic(trainingSet[0],trainingSet[1])

    for i in range(0,len(testingSet[1])):
        accuracy += abs(testingSet[1][i] - (testingSet[0][i] * params[0] + params[1]))/abs(testingSet[1][i])
    print(accuracy)
    return 1 - (accuracy/len(testingSet[1]))

def findMean (setData, setData2 = None):     #This will find the estimated mean of the data set
    #Formula: μ = Σ(Xi)/n , where μ is mean, Xi is a data point, n is the number of Xi elements
    mean = 0
    if setData2 is None:
        for i in setData:
            mean += i
        return mean/len(setData)

    else: #this assumes that the length of setData and setData2 have the same length
        for x,y in zip(setData,setData2):
            mean += x*y
        return mean/len(setData)


def findVariance(setData, mean = None): #This will find the estimated variance of the data set
    #Formula: σ^2 = (Σ(Xi - μ)^2) / (n-1) , where σ^2 is variance, μ is the mean of the set data, Xi is a data point, n is the number of Xi elements 

    if mean is None:
        mean = findMean(setData)
    variance = 0 
    for i in setData:
        variance += (i - mean)**2
    
    return variance/(len(setData) -1)

def findCorrelation(setX,setY):
    #Formula: Cov(X,Y) = E[XY] - E[X]E[Y]
    return findMean(setX,setY) - findMean(setX)*findMean(setY)

class linReg:
        
    def basic(setX,setY): #returns a list in the form [m,b] for y=mx+b
        #Formula: θm = (Σ((Xi - μx)*(Yi - μy)) / Σ(Xi -μx)^2 ) This is just Cov(x,y)/ var(x) just simplified
        #         θb = μy - θm*μx
        meanX = findMean(setX)
        meanY = findMean(setY)

        thetaM = 0
        tempX = 0
        for x,y in zip(setX,setY): #zip just combines the two arrays so i can traverse both at the same time
            thetaM += (x - meanX)*(y-meanY)
            tempX += (x - meanX)**2

        thetaM = thetaM/ tempX

        thetaB = meanY - thetaM*meanX

        return[thetaM, thetaB]
        
    def multiVar(dataSet,resultSet): 
        #dataSet will be 2D array (2xn matrix) with each row representing the data points of an independent variable. 
        #resultSet will be an 1D array (n-D vector) with the value that we are trying to predict.

        #LOGIC
        #   In this function we assume that the result set can be repesented by:
        #       Yi = θa + θbXi + θcZi ... + θnΦi + Wi, where Xi,Zi...,Φi are independent and Wi is ~ N(0,σ^2)
        #   We wil then use the most likely estimator and try to maximize the θ's by:
        #       min((Σ Yi -θa - θbXi - θcZi ... - θnΦi)^2) 
        #   The resulting matrix after derivation and setting to zero is:
        #       |1  μx      μz     ...   μΦ | |θa|   | μy  |
        #       |μx E[X^2]  E[XZ]  ... E[XΦ]| |θb|   |E[XY]|
        #       |μz E[XZ]   E[Z^2] ... E[ZΦ]| |θc| = |E[YZ]|
        #       |.   .       .           .  | | .|   |  .  |
        #       |.   .       .           .  | | .|   |  .  |
        #       |.   .       .           .  | | .|   |  .  |
        #       |μΦ E[XΦ]   E[ZΦ]  ...E[Φ^2]| |θΦ|   |E[YΦ]|
        #
        #   Solving this can be done using numpy. 
        
        #setting up the matrix and the Y-vector shown above
        matrix = []
        y_vector = []
        temp = []

        #first row setup
        for j in range(-1,len(dataSet)):
            if j == -1:
                temp.append(1)
            else:
                temp.append(findMean(dataSet[j]))

        y_vector.append(findMean(resultSet))

        matrix.append(temp)
        temp = []

        for i in range(len(dataSet)):
            for j in range(-1,len(dataSet)):
                if j == -1:
                    temp.append(findMean(dataSet[i]))
                else:
                    temp.append(findMean(dataSet[i],dataSet[j]))

            matrix.append(temp)
            temp = []
            y_vector.append(findMean(resultSet,dataSet[i]))

        return linalg.solve(matrix,y_vector)

    def nDegree(dataSet,resultSet,degree):
        #Formula: θm = Cov(g(x),y)/ var(g(x))
        #         θb = μy - θm*μ(g(x))
        #         Where g(x) is in the form: g(x) = x^k
        modifiedArr = []
        for i in dataSet:
            modifiedArr.append(i**degree)

        thetaM = findCorrelation(modifiedArr,resultSet) / findVariance(modifiedArr)

        thetaB = findMean(resultSet) - thetaM * findMean(modifiedArr)

        return [thetaM,thetaB]


if DEBUG :

    print("IN DEBUG MODE")
    print("CHECKING IF TEST SCRIPTS ARE IN SUB_DIRECTORY")

    if testScript:
        print("FOUND TESTING SCRIPTS")
        print("CREATING A SERIES OF BASIC TESTS")

        tests = []

        for i in range(1,TESTS +1):
            tests.append(turnToSets(line.lineSet.basic(i,i+2,N)))
            print('.',end = "")
        print('')

        print("TESTING BASIC LIN REG")
        for i in range(0,TESTS):
            acc = testBasicSets(tests[i],PORTION)
            print("Accuracy @",i, " is:", acc)
        #--------------------------------------------------------

        print('')
        print("CREATING A SERIES OF NORMAL TESTS")
        tests = []

        for i in range(1,TESTS+1):
            tests.append(turnToSets(line.lineSet.normal(i,i+2,N,STD_DEV)))
            print('.',end = "")
        print('')

        print("TESTING BASIC LIN REG")
        for i in range(0,TESTS):
            acc = testBasicSets(tests[i],PORTION)
            print("Accuracy @",i, " is:", acc)
        #--------------------------------------------------------

        print('')
        print("CREATING A SERIES OF CLUSTERED TESTS")
        tests = []

        for i in range(1,TESTS+1):
            tests.append(turnToSets(line.lineSet.clustered(i,i+2,N,STD_DEV,MEANX,VARX)))
            print('.',end = "")
        print('')
        print("TESTING BASIC LIN REG")
        for i in range(0,TESTS):
            acc = testBasicSets(tests[i],PORTION)
            print("Accuracy @",i, " is:", acc)

    else:
        print("CANNOT FIND SCRIPT. DEBUG WILL GO WITH HARDCODED VALUE")
        print("MEAN OF X IS: " , findMean(X))
        print("MEAN OF Y IS: " , findMean(Y))
        print("MEAN OF Z IS: " , findMean(Z))

        print("VARIANCE OF X IS: " , findVariance(X))
        print("VARIANCE OF Y IS: " , findVariance(Y))
        print("VARIANCE OF Y IS: " , findVariance(Z))

        temp = linReg.basic(X,Y)

        print("M OF BASIC LIN_REG IS: ",temp[0])
        print("B OF BASIC LIN_REG IS: ",temp[1])

        print("OUTPUTING THETA'S OF Y = A*X + B*Z + C")

        temp = linReg.multiVar([X,Z],Y)

        print(temp)

