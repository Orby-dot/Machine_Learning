#Linear regression model

#Purpose: To predict outcome of data under the assumption:
#   1: The observation is linearly correlated to the desired paramter
#   2: There is some noise W ~ N(0,u^2)
#   3: The model will be of the form: Y = a + bX where Y is the desired paramter and X is the observation

#Goal: To find a and b in the above equation

#Extensions: To allow any polynomial or have multiple variables correlated to the desired parameter

#Background info:
#   Refer to directory called "Background Info"

##WARNING
#   CORRELATION IS NOT CAUSATION. PLEASE KEEP THIS IN MIND WHEN YOU ARE USING THIS MODEL.
#Packages: Limited
import numpy
import argparse

#Debug variables
DEBUG = 1
X = [0,1,2,3,4,5,6,7,8,9,10]
Y = [0,2,4,6,8,10,12,14,16,18,20]

def findMean (setData):     #This will find the estimated mean of the data set
    #Formula: μ = Σ(Xi)/n , where μ is mean, Xi is a data point, n is the number of Xi elements
    mean = 0
    for i in setData:
        mean += i
    return mean/len(setData)

def findVariance(setData, mean = None): #This will find the estimated variance of the data set
    #Formula: σ^2 = (Σ(Xi - μ)^2) / (n-1) , where σ^2 is variance, μ is the mean of the set data, Xi is a data point, n is the number of Xi elements 

    if mean is None:
        mean = findMean(setData)
    variance = 0 
    for i in setData:
        variance += (i - mean)**2
    
    return variance/(len(setData) -1)


def linearRegBasic(setX,setY): #returns a list in the form [m,b] for y=mx+b
    #Formula: θm = (Σ((Xi - μx)*(Yi - μy)) / Σ(Xi -μx)^2 )
    #         θb = μy - θm*μx
    meanX = findMean(setX)
    meanY = findMean(setY)
    #varX = findVariance(setX,meanX)

    thetaM = 0
    tempX = 0
    for x,y in zip(setX,setY): #zip just combines the two arrays so i can traverse both at the same time
        thetaM += (x - meanX)*(y-meanY)
        tempX += (x - meanX)**2

    thetaM = thetaM/ tempX

    thetaB = meanY - thetaM*meanX

    return[thetaM, thetaB]
    

if DEBUG :
    print("IN DEBUG MODE")

    print("MEAN OF X IS: " , findMean(X))
    print("MEAN OF Y IS: " , findMean(Y))

    print("VARIANCE OF X IS: " , findVariance(X))
    print("VARIANCE OF Y IS: " , findVariance(Y))

    temp = linearRegBasic(X,Y)

    print("M OF LIN_REG IS: ",temp[0])
    print("B OF LIN_REG IS: ",temp[1])
