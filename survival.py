import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics



def sigmoid(z):
    hypothesis= 1.0/(1.0 + np.exp(-z))
    return hypothesis



def computeLrCost(x, y, theta):
    m= len(y)

    z= np.dot(x, theta.transpose())
    hypothesis= sigmoid(z)
    cost= (-y)*np.log(hypothesis)-(1-y)*np.log(1-hypothesis)
    costDerivative= np.dot(x.transpose(), hypothesis - y)/m

    return cost, costDerivative



def trainModel(x, y):
    m= len(y)
    alpha= 0.03
    iterations= 6000
    columns= np.shape(x)[1]
    theta= np.array([0]*columns)
    theta = theta.astype(np.float64)

    for i in range(iterations):
        cost, costDerivative= computeLrCost(x, y, theta)
        theta= theta - alpha*costDerivative

    return theta



def predictClass(x, theta, threshold):
    z= np.dot(x, theta.transpose())
    hypothesis= sigmoid(z)
    yPredicted= []

    for i in range(len(hypothesis)):
        if hypothesis[i] >= threshold:
            yPredicted.append(1)
        else:
            yPredicted.append(0)

    return yPredicted



def testPerformance(y, yPredicted):
    confusionMatrix= metrics.confusion_matrix(y, yPredicted)
    accuracy= accuracy_score(y, yPredicted)
    precision= precision_score(y, yPredicted)
    recall= recall_score(y, yPredicted)
    fScore= f1_score(y, yPredicted)

    return confusionMatrix, accuracy, precision, recall, fScore




# loading dataset
training_data= pd.read_csv("dataset.csv")
training_data= training_data.dropna()

x= training_data.iloc[:, 1:4].values
y= training_data.iloc[:, 4].values


# adding column of ones for x0
ones= [1 for i in range(len(y))]
x= np.column_stack((ones,x))


# split into train test sets
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25, random_state= 0)


# train
theta= trainModel(xTrain, yTrain)


# test
yPredicted= predictClass(xTest, theta, 0.5)
confusionMatrix, accuracy, precision, recall, fScore= testPerformance(yTest, yPredicted)


# print
print("Theta: ", theta)
print("Confusion matrix: ", confusionMatrix)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", fScore)
