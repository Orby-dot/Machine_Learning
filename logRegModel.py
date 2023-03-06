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

def calKFunc(params,setX,n = 0):
        #The K function is the deriviative of P(Xi) with respect of a, multipled by Xi ^ n
        #Formula: K(a,b) = Σ(-Xi^n)*(exp{-1(a+bXi)})*(1 + exp{-1(a+bXi)})^-2
        kResult = 0

        for x in setX:
            kResult += -1*x**n * m.exp(-1*(params[0]+params[1]*x)) * ((1 + m.exp(-1*params[0]+params[1]*x))**-2)

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
    return np.add(A,(1/temp)*(np.subtract(s,np.matmul(A,r)))* np.matmul(np.transpose(s),A))

class logReg:
    
    def basic (setX,setY):
        #Formula: a => 0 = ΣYi - P(Xi)
        #         b => 0 = ΣXi * (Yi - P(Xi))
        #I am going to use Broyden’s Method to find a and b

        #inital steps
        resultVector = [0,0] #placeholer
        pastVector = [0,1] #Going to set the inital values of a =0, and b = 1

        fVector = []
        pastFVector = [calFFunc(pastVector,setX,setY,0),calFFunc(pastVector,setX,setY,1)]

        aMatrix = []
        pastAMatrix = np.linalg.inv([ 
                        [calKFunc(pastVector,setX,0),calKFunc(pastVector,setX,1)],
                        [calKFunc(pastVector,setX,1),calKFunc(pastVector,setX,2)] ])
        buffer = np.matmul(pastAMatrix,pastFVector)
        resultVector = [pastVector[0] -buffer[0], pastVector[1] - buffer[1]]
        #looping begins
        while error(resultVector,pastVector) > 0.001:
            fVector = [calFFunc(resultVector,setX,setY,0),calFFunc(resultVector,setX,setY,1)]

          
            r = vectorSubtract(fVector,pastFVector)
            s = vectorSubtract(resultVector, pastVector)
            aMatrix = getNextMatrix(pastAMatrix,r,s)

            pastVector = resultVector
            pastAMatrix = aMatrix
            pastFVector = fVector
            buffer = np.matmul(pastAMatrix,pastFVector)
            resultVector = [pastVector[0] -buffer[0], pastVector[1] - buffer[1]]

        print(resultVector, " ", error(resultVector,pastVector))



             
             


