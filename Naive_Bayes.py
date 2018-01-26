import numpy as np
import statistics
import random
import csv
def class_prior(Y):
    classes = [[],[]]
    for i in range(len(Y)):
        if Y[i] not in classes[0]:
            classes[0].append(Y[i])
            classes[1].append(1)

        else:
            classes[1][classes[0].index(Y[i])] += 1
    print classes[0], classes[1]
    return [float(i)/len(Y) for i in classes[1]]


def proba1(X,Y,classes):
    proba = [[],[],[]]
    for i in range(len(Y)):
        if Y[i] not in proba[0]:
            proba[0].append(Y[i])
            proba[1].append(X[i])
            proba[2].append(sum([j for j in X[i]]))
        else:
            proba[1][proba[0].index(Y[i])] = np.array(proba[1][proba[0].index(Y[i])]) + np.array(X[i])
            proba[2][proba[0].index(Y[i])] += sum([f for f in X[i]])

    count = np.array(proba[2]) + np.array([len(classes) for i in range(len(classes))])
    last = []
    for j in range(len(classes)):
        a = float(1/float((proba[2][j]+len(X[0]))))
        b = [proba[1][j][k] + 1 for k in range(len(proba[1][j]))]
        last.append([(float(b[h]) * a) for h in range(len(X[0]))])
    return last

def naive_pro(prior,proba, x_new):
    naive = []
    for i in range(len(prior)):
        na = 1
        for j in range(len(x_new)):
            na *= (float(proba[i][j]))**x_new[j]
        naive.append(prior[i]*na)
    return naive.index(max(naive))

def accuracy(trainSet, labelTrain, testSet, labelTest):
    prior = class_prior(labelTrain)
    last = proba1(trainSet, labelTrain, prior)
    count = 0
    for i in range(len(testSet)):
        if (int(labelTest[i]) == naive_pro(prior, last, testSet[i])):
            count += 1
    return (count/float(len(testSet)) * 100)




d1 = np.array([2, 1, 1, 0, 0, 0, 0, 0, 0])
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N'])

#test data
d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)

    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    labelTrain = []
    for i in range(len(trainSet)):
        labelTrain.append(trainSet[i][-1])
        del trainSet[i][-1]

    labelTest = []
    for j in range(len(copy)):
        labelTest.append(copy[j][-1])
        del copy[j][-1]
    return [trainSet, labelTrain, copy, labelTest]

def main():
    filename = 'pima-indians-diabetes.dat.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, labelTrain, testSet, labelTest = splitDataset(dataset, splitRatio)
    print accuracy(trainingSet, labelTrain, testSet, labelTest)

main()