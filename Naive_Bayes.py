# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:13:35 2022

@author: user
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

# load the iris dataset
ds = pd.read_csv('C:/Users/user/Downloads/data.csv')

# store the feature matrix (X) and response vector (y)
X = ds.iloc[: , :26]
y = ds.Label


#Categorical labelling (Data Pre-processing)
label_encoder = preprocessing.LabelEncoder()

X['Address']= label_encoder.fit_transform(X['Address'])

X['CommandResponse']= label_encoder.fit_transform(X['CommandResponse'])

X['ControlMode']= label_encoder.fit_transform(X['ControlMode'])

X['ControlScheme']= label_encoder.fit_transform(X['ControlScheme'])

X['FunctionCode']= label_encoder.fit_transform(X['FunctionCode'])

X['InvalidDataLength']= label_encoder.fit_transform(X['InvalidDataLength'])

X['InvalidFunctionCode']= label_encoder.fit_transform(X['InvalidFunctionCode'])

X['PumpState']= label_encoder.fit_transform(X['PumpState'])

X['SolenoidState']= label_encoder.fit_transform(X['SolenoidState'])

#Standardization (Data Pre-processing)
scaler = StandardScaler()
scaler.fit(X)

X = scaler.fit_transform(X)

#Feature Selection
pca = PCA(n_components = 15)
pca.fit(X)
x_pca = pca.transform(X)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
			x_pca, y, test_size = 0.2)

# training the model on training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

def gnb_predict(sample):
    return gnb.predict(sample)

print("Accuracy : ", accuracy_score(y_pred,y_test))
print("Precision : ", precision_score(y_pred,y_test, average = 'weighted'))
print("Recall : ", recall_score(y_pred,y_test, average = 'weighted'))

"""
from knn import knn_predict

output=knn_predict(X_test[:1])

print(output)
"""