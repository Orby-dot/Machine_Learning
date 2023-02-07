#Linear regression model

#Purpose: To predict outcome of data under the assumption:
#   1: The observation is linearly correlated to the desired paramter
#   2: There is some noise W ~ N(0,σ^2)
#   3: The model will be of the form: Y = a + bX where Y is the desired paramter and X is the observation

#Goal: To find a and b in the above equation

#Extensions: To allow any polynomial or have multiple variables correlated to the desired parameter
#NOTE THIS IS IMMPLEMENTED

#Background info:
#   Refer to directory called "Background Info"

##WARNING
#   CORRELATION IS NOT CAUSATION. PLEASE KEEP THIS IN MIND WHEN YOU ARE USING THIS MODEL.
#Packages: Limited
from numpy import linalg
import numpy as np #<-- for debugging
import argparse #<-- might use for independent execution 
import matplotlib.pyplot as plt #<-- for testing only

#testing scripts for debugging
testScript = True
try:
    import scripts.lineCreation as line
except ImportError:
    testScript = False

#Debug variables
DEBUG = 1
HARDCODE = 0
N = 100000
TESTS = 1
STD_DEV = 5
MEANX = 0
VARX = 1000
PORTION = .5

VARNUM = 5
SAMPLEMAX = 10
SAMPLEMIN = -10

#If lineCreation.py cant be found i will use these values
X = [0,1,2,3,4,5,6]
Y = [15.5,23.5,12,4.5,-13,36,41]
Z = [5,7,3,1,-5,6,10]
A = [-1,-1,-4,5,6,10,8]
B = [3,2,1,8,9,-4,3]

def turnToSets(data):
    result = [[] for i in range(len(data[0]))]
    for i in range(0,len(data)):
        for k in range(0,len(data[i])):
            result[k].append(data[i][k])

    return result

#For basic lin reg testing
def testBasicSets(set,portion):
    accuracy = 0

    #creating training and testing sets
    trainingSet = []
    trainingSet.append(set[0][0:int(len(set[0])*portion)])
    trainingSet.append(set[1][0:int(len(set[1])*portion)])

    testingSet = []
    testingSet.append(set[0][int(len(set[0])*portion):])
    testingSet.append(set[1][int(len(set[1])*portion):])

    #Calculates the params of the training set
    params = linReg.basic(trainingSet[0],trainingSet[1])

    #finds absolute % error
    for i in range(0,len(testingSet[1])):
        accuracy += abs(testingSet[1][i] - (testingSet[0][i] * params[0] + params[1]))/abs(testingSet[1][i])
    return 1 - (accuracy/len(testingSet[1]))

#My testing function on multi var :D
def testMultiSet(set,portion):
    accuracy = 0
    trainingSet = []
    testingSet = []

    #Creates testing and training sets
    for i in range(0,len(set)):
        trainingSet.append(set[i][0:int(len(set[i])*portion)])
        testingSet.append(set[i][int(len(set[i])*portion):])

    #Calculates the params of the training set
    calParams = linReg.multiVar(trainingSet[0:len(trainingSet)-1],trainingSet[-1])

    #finds absolute % error
    for i in range(0,len(testingSet[-1])):
        #I hate pass by object reference
        calYValue = calParams[:]
        calYValue = int(calYValue[0])

        #Gets the cal-ed Y value
        for k in range(0,len(testingSet)-1):
            calYValue += calParams[k+1] * testingSet[k][i]

        accuracy += abs(testingSet[-1][i] - calYValue)/abs(testingSet[-1][i])

    return 1 - (accuracy/len(testingSet[0]))



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
        #YEAH YOU DID MATH WRONG LOL FIX IT 
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

    if testScript and HARDCODE !=1:
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
        #--------------------------------------------------------

        print('')
        print("CREATING A SERIES OF BASIC MULTIVARABLE TESTS")
        tests = []

        #creating points
        for i in range(1,TESTS+1):

            #generatating random starting params
            params = [(SAMPLEMAX - SAMPLEMIN)*np.random.random_sample() + SAMPLEMIN]
            inital = []
            for j in range(0,VARNUM):
                params.append((SAMPLEMAX - SAMPLEMIN)*np.random.random_sample() + SAMPLEMIN)
                inital.append ((SAMPLEMAX - SAMPLEMIN)*np.random.random_sample() + SAMPLEMIN)

            tests.append(turnToSets(line.multiLineSet.basic(params,inital,N,STD_DEV)))
            print('.',end = "")

        print('')
        print("TESTING MULTI LIN REG")

        for i in range(0,TESTS):
            acc = testMultiSet(tests[i],PORTION)
            print("Accuracy @",i, " is:", acc)

        #--------------------------------------------------------

        print('')
        print("CREATING A SERIES OF STEP MULTIVARABLE TESTS")
        tests = []

        #creating points
        for i in range(1,TESTS+1):

            #generatating random starting params
            params = [(SAMPLEMAX - SAMPLEMIN)*np.random.random_sample() + SAMPLEMIN]
            inital = []
            step = []
            for j in range(0,VARNUM):
                params.append((SAMPLEMAX - SAMPLEMIN)*np.random.random_sample() + SAMPLEMIN)
                inital.append ((SAMPLEMAX - SAMPLEMIN)*np.random.random_sample() + SAMPLEMIN)
                step.append ((SAMPLEMAX - SAMPLEMIN)*np.random.random_sample() + SAMPLEMIN)

            tests.append(turnToSets(line.multiLineSet.step(params,inital,step,N,STD_DEV)))
            print('.',end = "")

        print('')
        print("TESTING MULTI LIN REG")

        for i in range(0,TESTS):
            acc = testMultiSet(tests[i],PORTION)
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
        ah = []
        for i in range(0,len(X)):
            ah.append(X[i] + 3*Z[i] + .5*A[i] - B[i] + 4)

        temp = linReg.multiVar([X,Z,A,B],ah)

        print(temp)

