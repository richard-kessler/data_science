# -*- coding: utf-8 -*-
"""
******************************************************************************************************************************

Project: Network Intrusion Detection using Artificial Neural Network(ANN) Discovery

Program Name: PYTHON_NID_ANN_12052019.01.00

Author: Richard Kessler 

Date Created: 12.05.2019

Purpose: ANN analysis of Network Intrusion data to identify feature selection and model parameters.  

Data Scope: Network Intrusion training data which contain network logs and traffic features used to detect anomalies.

******************************************************************************************************************************
 
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Importing the dataset
dataset = pd.read_csv('Train_data.csv')
X = dataset.iloc[:, 1:42].values
y = dataset.iloc[:, 0].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encode Categorical Variable 1
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
#Encode Categorical Variable 2
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
#Encode Categorical Variable 3
labelencoder_X_3 = LabelEncoder()
X[:, 2] = labelencoder_X_3.fit_transform(X[:, 2])


#Create Dummy Variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#Removing the first dummy variable as 3 were created and we need to reduce to 2
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer with Dropout
classifier.add(Dense(activation = 'relu', units = 22, kernel_initializer = 'uniform', input_dim = 42))
classifier.add(Dropout(rate = 0.1))

#Adding a second hidden layer
classifier.add(Dense(activation = 'relu', units = 22, kernel_initializer = 'uniform'))
classifier.add(Dropout(rate = 0.1))

#Adding an output layer (binary outcome)
classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 

# Confusion Matrix Accuracy (99% Accurate)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Evaluate the ANN using Kfold - Pre-Eval Accuracy: 99.4%, Post-Eval Accuracy: 99.63
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', units = 22, kernel_initializer = 'uniform', input_dim = 42))
    classifier.add(Dense(activation = 'relu', units = 22, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variace = accuracies.std()


#Tuning the ANN to discover the best parameters for Keras optimization
 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', units = 22, kernel_initializer = 'uniform', input_dim = 42))
    classifier.add(Dense(activation = 'relu', units = 22, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
#GridSearch Dictionary
parameters = {'batch_size': [25, 32], 
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


"""

********************************************************************************************************************************

Analysis Outcome: Overall, the ANN model performed extremely well with predictions coming in over 99% using the initial training
data without feature modification. Although this model resulted in high accuracy...it shows preliminary expectations of overfitting.
Recommended Actions: Significantly reduce feature scope and re-validate which I believe will result in <99% accuracy but a less 
biased model. 

******************************************************************************************************************************** 


