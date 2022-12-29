#Linear regression model

#Purpose: To predict outcome of data under the assumption:
#   1: The observation is linearly correlated to the desired paramter
#   2: There is some noise W ~ N(0,u^2)
#   3: The model will be of the form: Y = a + bX where Y is the desired paramter and X is the observation

#Goal: To find a and b in the above equation

#Extensions: To allow any polynomial or have multiple variables correlated to the desired parameter

#Background info:
#   "Add youtube videos here"
#   "Add textbook link here"

##WARNING
#   CORRELATION IS NOT CAUSATION. PLEASE KEEP THIS IN MIND WHEN YOU ARE USING THIS MODEL.
#Packages: Limited
import numpy
import argparse

