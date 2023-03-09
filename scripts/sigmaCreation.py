#Sigma Creation Script

#Purpose: To create a set of (x,y) points that models an ideal sigma function

#Goal: To test logical regression models

#Extensions: Allow the creation of multivariable sigma function

#Background info: Refer to the wiki link in the folder
#Turns a set of (x,y) points to a cvs file 
import math as m
from numpy import random
def toFile(data, name = None):
    if name is None:
        file = open("result.csv", "w")
    else:
        file =open(name +".csv", "w")

    for i in data:
        for j in i:
            file.write(str(j) + ",")

        file.write("\n")
    return 0

def calPFunc(params,x):
    #Formula: (1+exp{-(a + bX)})^(-1)
    return (1 + m.exp(-1*(params[0]+params[1]*x)))**-1

def getY(a,b,x):
    rand = random.random_sample()
    p = calPFunc([a,b],x)
    if(rand > p ):

        return 0
    else:
        return 1


class sigmaSet:
    def basic(a,b,n,start,end):
    #Function: Given a and b create a series of n evenly spaced out points from ranges[start,end]
        increment = (end-start)/n
        result = []
        for i in range(0,n):
            result.append([start + (i)*increment,getY(a,b,start + (i)*increment)])
        return result

toFile(sigmaSet.basic(-6.25,1.25,100,0,10))