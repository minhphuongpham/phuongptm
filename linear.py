#predict age base on height and weight by using linear regression methods

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import loadtxt
import sympy

data = open('static.txt','r')
lines= data.readlines()
#X=np.array([50,60,48,75,69,66,77,88])
#Y=np.array([80,88,89,99,77,88,120,111])
#y=np.array([141,145,156,165,176,157,180,177])
X=[]
Y=[]
y=[]
for line in lines:
    new = line.split(' ')
    X.append(float(str(new[1])))
    Y.append(float(str(new[2])))
    y.append(float(str(new[0])))
X=np.array(X)
Y=np.array(Y)
y=np.array(y)
A= np.array([X,Y]).reshape(len(X),2)
one= np.ones((X.shape[0],1))
Abar = np.concatenate((one,A),axis=1)
print (Abar)
B= np.dot(Abar.T,Abar)
b = np.dot(Abar.T,y)
w = np.dot(np.linalg.pinv(B),b)
print('w = ', w)
w_0=w[0]
w_1=w[1]
w_2=w[2]

z= w_0 + w_1*X + w_2*Y
y1 = np.array(range(len(X)))
plt.plot(y1,y, color = 'blue', label = 'Initial data')
plt.plot(y1,z, color = 'red', label = 'Predict data')
plt.show()

#print (z)
#Because of so much noise ages, the linear regression method does not give accurate predictions



