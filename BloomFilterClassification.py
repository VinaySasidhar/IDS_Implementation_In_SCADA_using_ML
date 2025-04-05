import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#Reading the dataset

ds = pd.read_csv('C:/Users/user/Downloads/data.csv')

X_Train_Bloom_Filter = ds.loc[ds['Label'] == 'Good']



#Categorical labelling (Data Pre-processing)
label_encoder = preprocessing.LabelEncoder()

############BLOOM FILTER CLASSIFICATON##########################


X_Train_Bloom_Filter.drop(['Label'],axis=1,inplace=True)

X_Train_Bloom_Filter['Address']= label_encoder.fit_transform(X_Train_Bloom_Filter['Address'])

X_Train_Bloom_Filter['CommandResponse']= label_encoder.fit_transform(X_Train_Bloom_Filter['CommandResponse'])

X_Train_Bloom_Filter['ControlMode']= label_encoder.fit_transform(X_Train_Bloom_Filter['ControlMode'])

X_Train_Bloom_Filter['ControlScheme']= label_encoder.fit_transform(X_Train_Bloom_Filter['ControlScheme'])

X_Train_Bloom_Filter['FunctionCode']= label_encoder.fit_transform(X_Train_Bloom_Filter['FunctionCode'])

X_Train_Bloom_Filter['InvalidDataLength']= label_encoder.fit_transform(X_Train_Bloom_Filter['InvalidDataLength'])

X_Train_Bloom_Filter['InvalidFunctionCode']= label_encoder.fit_transform(X_Train_Bloom_Filter['InvalidFunctionCode'])

X_Train_Bloom_Filter['PumpState']= label_encoder.fit_transform(X_Train_Bloom_Filter['PumpState'])

X_Train_Bloom_Filter['SolenoidState']= label_encoder.fit_transform(X_Train_Bloom_Filter['SolenoidState'])

print(X_Train_Bloom_Filter)
#Standardization (Data Pre-processing)

scaler_BF = StandardScaler()
scaler_BF.fit(X_Train_Bloom_Filter)

X_Train_Bloom_Filter = scaler_BF.fit_transform(X_Train_Bloom_Filter)
print(X_Train_Bloom_Filter)

#Feature Selection
pca_BF = PCA(n_components = 15)
pca_BF.fit(X_Train_Bloom_Filter)
x_pca_BF = pca_BF.transform(X_Train_Bloom_Filter)
  
print(x_pca_BF.shape)


from bloomfilter import BloomFilter

n = 84226 #no of items to add
p = 0.05 #false positive probability

bloomf = BloomFilter(n,p)
print("Size of bit array:{}".format(bloomf.size))
print("False positive Probability:{}".format(bloomf.fp_prob))
print("Number of hash functions:{}".format(bloomf.hash_count))

for item in x_pca_BF:
	bloomf.add(item)
    

def BloomFilterCheck(sample):
    return bloomf.check(sample)


"""
if bloomf.check(item):
    print("The data is probably an anomaly")
else:
    print("The data is definetly an anomaly")
"""