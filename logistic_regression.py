# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 16:25:00 2022

@author: user
"""

# Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Loading data
ds = pd.read_csv('C:/Users/user/Downloads/data.csv')

# Create feature and target arrays
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

lr = LogisticRegression(random_state = 0)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

def lr_predict(sample):
    return lr.predict(sample)

print("Accuracy : ", accuracy_score(y_pred,y_test))
print("Precision : ", precision_score(y_pred,y_test, average = 'weighted'))
print("Recall : ", recall_score(y_pred,y_test, average = 'weighted'))

