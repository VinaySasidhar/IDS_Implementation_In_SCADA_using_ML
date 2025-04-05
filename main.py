# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:25:09 2022

@author: Vinay
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from BloomFilterClassification import BloomFilterCheck
from voting_classifier import Hardvoting_pred
from voting_classifier import Softvoting_pred

#Reading the dataset

ds = pd.read_csv('C:/Users/user/Downloads/data.csv')


sample_index = len(ds)
ds.loc[sample_index] =['0x0', 'X', 'X',	'X', 0, 123, '0x0', 'InvalidDataLength', 'InvalidFunctionCode', 0, 0, 0, 0, 0, 0, 'X', 0, 'X', 125, 0, 0, 0, 0, 0, 0, 0, 'DOS']


ds.drop(['Label'],axis=1,inplace=True)

#Categorical labelling (Data Pre-processing)
label_encoder = preprocessing.LabelEncoder()

ds['Address']= label_encoder.fit_transform(ds['Address'])

ds['CommandResponse']= label_encoder.fit_transform(ds['CommandResponse'])

ds['ControlMode']= label_encoder.fit_transform(ds['ControlMode'])

ds['ControlScheme']= label_encoder.fit_transform(ds['ControlScheme'])

ds['FunctionCode']= label_encoder.fit_transform(ds['FunctionCode'])

ds['InvalidDataLength']= label_encoder.fit_transform(ds['InvalidDataLength'])

ds['InvalidFunctionCode']= label_encoder.fit_transform(ds['InvalidFunctionCode'])

ds['PumpState']= label_encoder.fit_transform(ds['PumpState'])

ds['SolenoidState']= label_encoder.fit_transform(ds['SolenoidState'])


#Standardization (Data Pre-processing)

scaler = StandardScaler()
scaler.fit(ds)

ds = scaler.fit_transform(ds)

#Feature Selection
pca = PCA(n_components = 15)
pca.fit(ds)
x_pca = pca.transform(ds)

if BloomFilterCheck(x_pca[sample_index:]):
    print("\nThe data is probably an anomaly")
    print("HardVoting Prediction:")
    print(Hardvoting_pred(x_pca[sample_index]))
    print("SoftVoting Prediction:")
    print(Softvoting_pred(x_pca[sample_index]))
else:
    print("\nThe data is definetly an anomaly")
  