# -*- coding: utf-8 -*-
"""Breast_cancer_SLP.ipynb

# Loading DataSets
"""

import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

breast_cancer = sklearn.datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

print(X)
print(Y)

print(X.shape, Y.shape)

import pandas as pd

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)

data['class'] = breast_cancer.target

data.head()

data.describe()

print(data['class'].value_counts())

print(breast_cancer.target_names)

data.groupby('class').mean()



"""# Train - Test Split"""

from sklearn.model_selection import train_test_split

X = data.drop('class', axis = 1)
Y = data['class']

type(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

print(X.shape, X_train.shape, X_test.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

print(X.shape, X_train.shape, X_test.shape)

print(Y.mean(), Y_train.mean(), Y_test.mean())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y) # Stratify is implied to get similar values in train, test and mean

print(X_train.mean(), X_test.mean(), X.mean())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1) # Random state is applied so that the loop runs with same values everytime

Y_test.head()

print(X_train.mean(), X_test.mean(), X.mean())

"""# Perceptron Class

![download.jfif](attachment:download.jfif)

$y = 1, \mbox{if} \sum_i w_i x_i >= b$

$y = 0, \mbox{otherwise}$
"""

class Perceptron:

  def __init__(self):
    self.w = None
    self.b = None

  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0

  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return Y

  def fit(self, X, Y):

    self.w = np.ones(X.shape[1])
    self.b = 0

    for x, y in zip(X, Y):
      y_pred = self.model(x)
      if y == 1 and y_pred == 0:
        self.w = self.w + x
        self.b = self.b + 1
      elif y == 0 and y_pred == 1:
        self.w = self.w - x
        self.b = self.b - 1

perceptron = Perceptron()

X_train = X_train.values
X_test = X_test.values

perceptron.fit(X_train, Y_train)

plt.plot(perceptron.w)
plt.show()

"""# Metrics"""

from sklearn.metrics import accuracy_score

y_pred_train = perceptron.predict(X_train)
print(accuracy_score(y_pred_train, Y_train))

print(X_test.shape)
y_pred_test = perceptron.predict(X_test)
print(accuracy_score(y_pred_test, Y_test))

"""# Model with iteration"""

""" Epochs are going through the data over and over from first or from where it is assigned."""
""" Learning rate is added to the parameter multiplication here"""
""" Both Epochs and Learning rate are called as Hyper parameters which works in conjunction """

class Perceptron_epoch:

  def __init__(self):
    self.w = None
    self.b = None

  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0

  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return Y

  def fit(self, X, Y, epochs = 50, lr = 0.1):

    self.w = np.ones(X.shape[1])
    self.b = 0

    wt_matrix = []    # Matrix of weights

    accuracy = {}
    max_accuracy = 0

    for i in range(epochs):
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        if y == 1 and y_pred == 0:
          self.w = self.w + lr * x    # lr is the learning rate
          self.b = self.b + lr * 1
        elif y == 0 and y_pred == 1:
          self.w = self.w - lr * x
          self.b = self.b - lr * 1

      wt_matrix.append(self.w)

      accuracy[i] = accuracy_score(self.predict(X),Y)
      if (accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        chkptw = self.w     # Checkpoints for w and b (Parameters) corresponding to highest accuracy.
        chkptb = self.b     # Checkpoint is done to capture the optimal parameters of (w, b) calculated during training set with highest accuracy.

    self.w = chkptw    # Assigning checkpointed w and b with highest accuracy model.
    self.b = chkptb

    print(max_accuracy)
    plt.plot(list(accuracy.values()))
    plt.ylim([0,1])
    plt.grid()
    plt.show()

    return np.array(wt_matrix), accuracy

perceptron = Perceptron_epoch()

wt_matrix = perceptron.fit(X_train, Y_train, 10000)     # X_train, Y_train, epoch and Learning rate.

Y_pred_train = perceptron.predict(X_train)
print(accuracy_score(Y_pred_train, Y_train))

plt.plot(perceptron.w)
plt.show()

def fit(self, X, Y, epochs=50, lr=0.1):

    self.w = np.ones(X.shape[1])
    self.b = 0

    wt_matrix = []  # Matrix of weights

    accuracy = {}

    for i in range(epochs):
        for x, y in zip(X, Y):
            y_pred = self.model(x)
            if y == 1 and y_pred == 0:
                self.w = self.w + lr * x  # lr is the learning rate
                self.b = self.b + lr * 1
            elif y == 0 and y_pred == 1:
                self.w = self.w - lr * x
                self.b = self.b - lr * 1

        wt_matrix.append(self.w)

        accuracy[i] = accuracy_score(self.predict(X), Y)

    return np.array(wt_matrix), accuracy

perceptron = Perceptron_epoch()
wt_matrix, accuracy = perceptron.fit(X_train, Y_train, 10000)
Y_pred_train = perceptron.predict(X_train)
print(accuracy_score(Y_pred_train, Y_train))

# Find the epoch with the highest accuracy
max_accuracy = 0
best_epoch = 0
for epoch, accuracy in accuracy.items():
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_epoch = epoch

print("Best number of epochs:", best_epoch)
print("Maximum accuracy:", max_accuracy)

