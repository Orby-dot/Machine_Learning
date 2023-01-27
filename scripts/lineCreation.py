#Line Creation Script

#Purpose: To create a set of (x,y) points that fit one of two scenarios:
#   1. The points exactly match y = mx +b
#   2. The points approximately match y= mx + b + Wi, with the additional term Wi ~ N(0,σ^2)

#Goal: To test linear regression models

#Extensions: Allow the creation of multivariable linear equations (e.g. Yi = θa + θbXi + θcZi ... + θnΦi + Wi)

#Background info: Refer to linear equations and the simple line equation from highschool

#Packages: Limited
import argparse
from numpy import random

#Turns a set of (x,y) points to a cvs file 
def toFile(data, name = None):
    if name is None:
        file = open("result.csv", "w")
    else:
        file =open(name +".csv", "w")

    for i in data:
        file.write(i[0]+','+i[1] + '\n')
    return

class lineSet:
    def basic(m,b,n):
        #Function: Creates a 2D array that holds "n" pairs of evenly spaced (x,y) points with x's range [0,n-1] that direct map y= mx + b
        result = []
        for i in range(0,n):
            point = [i,i*m + b]
            result.append(point)
        return result

    def normal(m,b,n,std_dev):
        #Function: Creates a 2D array that holds "n" pairs of evenly spaced (x,y) points with x's range [0,n-1] that  map y= mx + b + Wi
        #   Wi is i.i.d and Wi ~ N(0,σ^2)
        result =[]
        for i in range(0,n):
            point = [i,i*m + b + random.normal(0,std_dev,1)]
            result.append(point)
        return result

    def clustered(m,b,n,std_dev,meanX,varX):
        #Function: Creates a 2D array that holds "n" pairs of clustered (x,y) points with X ~(μx,σx^2) and
        #          Y = mX + b + Wi, with Wi is i.i.d and Wi~ N(0,σ^2)
        result =[]
        for i in range(0,n):
            x = random.normal(meanX,varX**.5,1)
            point = [x,x*m + b + random.normal(0,std_dev,1)]
            result.append(point)
        return result

class multiLineSet:
    def basic(params,inital,n,std_dev = 0):
        #Function: Creates a 2D array that holds "n" pairs of evenly spaced (x,..Φi,y) points
        # with x's range [inital[0],inital[0] + n]... and Φi's range [inital[k],inital[k] + n]
        # that directly map Yi = params[0] + params[1]Xi +  ... + params[k]Φi + Wi
        result = []
        for i in range(0,n):
            point = []
            yValue = params[0]
            for k in range( 0,len(inital)):
                point.append(inital[k]+i)
                yValue += (inital[k]+i)*params[k+1]
            point.append(yValue + random.normal(0,std_dev,1))

            result.append(point)
        return result

    def step(params,inital,steps,n,std_dev = 0):
         #Function: Creates a 2D array that holds "n" pairs of evenly spaced (x,..Φi,y) points
         # with x's range [inital[0],inital[0] + n*steps[0]]... and Φi's range [inital[k],inital[k] + n*steps[k]]
         #  that map Yi = params[0] + params[1]Xi +  ... + params[k]Φi + Wi
        result = []
        for i in range(0,n):
            point = []
            yValue = params[0]
            for k in range( 0,len(inital)):
                point.append(inital[k]+i*steps[k])
                yValue += (inital[k]+i*steps[k])*params[k+1]
            point.append(yValue + random.normal(0,std_dev,1))

            result.append(point)
        print(result[0])
        return result


    def clustered(params,mean,var,n,std_dev = 0):
        #Function: Creates a 2D array that holds "n" pairs of clustered (x,z,y) points with X ~(μx,σx^2) and Z ~(μz,σz^2)
        #          Y = aX + bZ + c + Wi, with Wi is i.i.d and Wi~ N(0,σ^2)
        result =[]
        for i in range(0,n):
            point = []
            yValue = params[0]
            for k in range (0,len(mean)):
                temp = random.normal(mean[k],var[k]**.5,1)
                point.append(temp)
                yValue += (temp)*params[k+1]
            point.append(yValue + random.normal(0,std_dev,1))
            result.append(point)
        return result



