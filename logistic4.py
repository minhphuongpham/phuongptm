#predict sex base on weight and height by using logistic regression
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

#P = np.array([[0.5,0.75,1,1.25,1.5,1.75,1.75,2,2.25,2.5,2.75,3,3.25,3.5,4,4.25,4.5,4.75,5,5.5]])
#y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])
#print(P.shape[1])
#Q = np.concatenate((np.ones((0, P.shape[1])), P), axis = 0)
#print (Q)
#extend data
data = open('static.txt','r')
lines= data.readlines()
Z=[]
Y=[]
y=[]
for line in lines:
    new = line.split(' ')
    Z.append(float(str(new[2])))
    Y.append(float(str(new[3])))
    if (new[0] == 'm'):
        y.append(1)
    else:
        y.append(0)
Z=np.array(Z)
Y=np.array(Y)
y=np.array(y)
one= np.ones((1,Z.shape[0]))
#print (one)
#X = np.concatenate((one,A),axis = 1)
A = np.array([Z,Y])
X = np.concatenate((one,A),axis =0)
#print (X)
def sigmoid(s):
    return 1/(1+np.exp(-s))
def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        #mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:,i].reshape(d,1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T,xi))
            w_new = w[-1] + eta*(yi-zi)*xi
            count +=1

            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

eta = 0.05
d = X.shape[0]
w_init = np.random.randn(d,1)
#print (w_init)

w = logistic_sigmoid_regression(X,y,w_init,eta)
print (w[-1])
Ages = sigmoid(w[-1][0]+w[-1][1]*Z+w[-1][2]*Y)
#print (Ages)
y0 = np.array([i for i in range(len(Z))])
fig, ax = plt.subplots(1,3)
ax[0].plot(y0,y,'ro', color = 'blue', label = 'Initial data')
ax[0].plot(y0,Ages,'ro', color = 'red', label = 'Predict data')
ax[1].plot(y0,y,'ro', color = 'blue', label = 'Initial data')
ax[2].plot(y0,Ages,'ro', label = 'Predict data')
ax[0].set_title('Initial data and Predict data')
plt.show()
