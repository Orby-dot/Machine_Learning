#Logical regression model

#Purpose: To predict outcome of data under the assumption:
#   1: Theresult can be represented as a binary outcome (0 and 1 for my model)
#   2: The cumulative probability model is in the form: Yi = (1 + exp{-(X-u)/s})^(-1)
#       Where "u" is the point where the probability is 50% and "s" is the span
#       This can be rewriten in the form:
#           Yi = (1+exp{-(a + bX)})^(-1)
#           Where a = -u/s and b = 1/s
#Goal: To find a and b in the above equation

#Extensions: To allow any polynomial or have multiple variables correlated to the desired parameter


#Background info:
#   Refer to directory called "Background Info"
from numpy import linalg
import numpy as np #<-- for debugging
import argparse #<-- might use for independent execution 
import matplotlib.pyplot as plt #<-- for testing only
import math as m


#testing scripts for debugging
testScript = True
try:
    import scripts.sigmaCreation as sigma
except ImportError:
    testScript = False

X = [0.5,.75,1,1.25,1.5,1.75,1.75,2,2.25,2.5,2.75,3,3.25,3.50,4,4.25,4.5,4.75,5,5.5]
Y = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]

def initPrediction(setX,setY):
    #Given the raw data make a crude guess on the parameters of the sigma function

    #S can either be -1,0,1
    #   -1: if more 1's appear earlier on
    #   0: if evenly spread out
    #   1: if more 1's later on
    s =0;
    u=0;
    numOfOnesE =0
    numOfOnesL=0

    earliestZero = len(setY)-1
    latestZero = 0
    earliestOne = len(setY)-1
    latestOne = 0

    for i in range(len(setY)):

        if i < len(setY)/2 and setY[i] == 1:
            numOfOnesE +=1
        elif i >= len(setY)/2 and setY[i] == 1:
            numOfOnesL +=1

        if setY[i] == 0:
            earliestZero = i if (i < earliestZero) else earliestZero
            latestZero = i if (i > latestZero) else latestZero

        elif setY[i] ==1:
            earliestOne = i if (i < earliestOne) else earliestOne
            latestOne = i if (i > latestOne) else latestOne


    if numOfOnesE> numOfOnesL:
        s = -1
        u = (setX[earliestZero] + setX[latestOne])/2
    elif numOfOnesE < numOfOnesL:
        s = 1
        u = (setX[latestZero] + setX[earliestOne])/2

    else:
        u = (setX[0] + setX[-1])/2


    return [(-1*u/s),1/s]

def turnToSets(data):
    result = [[] for i in range(len(data[0]))]
    for i in range(0,len(data)):
        for k in range(0,len(data[i])):
            result[k].append(data[i][k])

    return result

def calKFunc(params,setX,n = 0):
        #The K function is the deriviative of P(Xi) with respect of a, multipled by Xi ^ n
        #Formula: K(a,b) = Σ(-Xi^n)*(exp{-1(a+bXi)})*(1 + exp{-1(a+bXi)})^-2
        kResult = 0

        for x in setX:
            kResult += -1*(x**n) * m.exp(params[0]+params[1]*x) * ((1 + m.exp(params[0]+params[1]*x))**-2)

        return kResult

def calPFunc(params,x):
    #Formula: (1+exp{-(a + bX)})^(-1)
    return (1 + m.exp(-1*(params[0]+params[1]*x)))**-1

def calFFunc(params,setX,setY,n=0):
    #Formula: Σ(Yi - P(Xi)) *Xi^n
    result = 0

    for i in range(len(setX)):
        result += (setX[i] ** n) *(setY[i]- calPFunc(params,setX[i]))

    return result

def error(vector,pastVector):
    return abs(vector[0]-pastVector[0]) + abs(vector[1]-pastVector[1]) 

def vectorSubtract(x,y):
    result =[];
    for i in range(len(x)):
        result.append(x[i]-y[i])

    return result

def getNextMatrix(A,r,s):
    temp = np.matmul(A,r)
    temp = np.matmul(np.transpose(s),temp)

    return np.add(A,(1/temp)*(np.outer(np.subtract(s,np.matmul(A,r)), np.matmul(np.transpose(s),A))))

class logReg:
    
    def basic (setX,setY):
        # f = open("Debug.txt","w")
        #Formula: a => 0 = ΣYi - P(Xi)
        #         b => 0 = ΣXi * (Yi - P(Xi))
        #I am going to use Broyden’s Method to find a and b

        #inital steps
        resultVector = [0,0] #placeholer
        pastVector = initPrediction(setX,setY) #Going to set the inital values of a =0, and b = 1
        print("PAST VECTOR:",pastVector)

        fVector = []
        pastFVector = [calFFunc(pastVector,setX,setY,0),calFFunc(pastVector,setX,setY,1)]
        # print("PAST F VECTOR",pastFVector)
        aMatrix = []
        pastAMatrix = np.linalg.inv([ 
                        [calKFunc(pastVector,setX,0),calKFunc(pastVector,setX,1)],
                        [calKFunc(pastVector,setX,1),calKFunc(pastVector,setX,2)] ])
        
        # f.write("PAST A {}\n".format(pastAMatrix))
        buffer = np.matmul(pastAMatrix,np.transpose(pastFVector))
        resultVector = np.subtract(pastVector,buffer)
        #looping begins
        # f.write("INIT RESULT IS {} ".format(resultVector))
        while error(resultVector,pastVector) > 0.0000001:

            fVector = [calFFunc(resultVector,setX,setY,0),calFFunc(resultVector,setX,setY,1)]
            r = np.subtract(fVector,pastFVector)
            s = np.subtract(resultVector,pastVector)
            aMatrix = getNextMatrix(pastAMatrix,r,s)

            pastVector = resultVector[:]
            pastAMatrix = aMatrix[:]
            pastFVector = fVector[:]
            buffer = np.matmul(aMatrix,np.transpose(fVector))
            # print(buffer)
            resultVector = np.subtract(resultVector,buffer)

            # print("ERROR", error(resultVector,pastVector))
            print("F VECTOR",fVector[0], " ", fVector[1])
            print("RESULT VECTOR",resultVector, " ", error(resultVector,pastVector))
            # f.write("ERROR {}\n".format(error(resultVector,pastVector)))
            # f.write("F VECTOR {} {}\n".format(fVector[0],fVector[1]))
            # f.write("RESULT VECTOR{}\n".format(resultVector))
            # f.write("----------------\n")
        print(resultVector, " ", error(resultVector,pastVector))

temp = turnToSets(sigma.sigmaSet.basic(-5.657,6.429,100000,-100,100))
logReg.basic(temp[0],temp[1])


             
             


