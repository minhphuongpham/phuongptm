import numpy as np
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt


#Create four populations of 100 observations each
pop1_X, pop1_Y = make_regression(n_samples=100, noise=20, n_informative=1,
                                 n_features=1, random_state=1, bias=0)
pop2_X, pop2_Y = make_regression(n_samples=100, noise=20, n_informative=1,
                                 n_features=1, random_state=1, bias=100)
pop3_X, pop3_Y = make_regression(n_samples=100, noise=20, n_informative=1,
                                 n_features=1, random_state=1, bias=-100)

#Stack them together
pop_X = np.concatenate((pop1_X, pop2_X, pop3_X))
pop_Y = np.concatenate((pop1_Y, 2 * pop2_Y, -2 * pop3_Y))

# Add intercept to X
pop_X = np.append(pop_X, np.vstack(np.ones(len(pop_X))),1)

pop_Y = np.vstack(pop_Y)

## parameters
n_learning_rate = 0.1

## Specify prediction function
def fx(theta, X):
    return np.dot(X, theta)

## Specify cost function
def fcost(theta, X, y):
    return (1./2*len(X)) * np.sum((np.vstack(fx(theta,X)) - y)**2)

def gradient(theta, X, y):
    grad_theta = (1./len(X)) * np.multiply((fx(theta, X)) - y, X)
    return grad_theta

## Do stochastic gradient descent
# starting values for alpha and beta
theta = [0,0]

#record starting theta and cost
arraytheta = np.array([theta])
arraycost = np.array([])

#feed data through and update theta, capture cost and theta history
for i in range(2, len(pop_X)):
    # calculate cost for theta on current point
    cost = fcost(theta, pop_X[0:i], pop_Y[0:i])
    arraycost = np.append(arraycost, cost)
    # update theta with gradient descent
    theta = theta - n_learning_rate * gradient(theta, pop_X[i], pop_Y[i])
    arraytheta = np.vstack([arraytheta, theta])

print theta