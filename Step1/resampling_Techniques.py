import os
import sys
import argparse
import pandas as pd
import numpy as np
import xlwt

from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE,KMeansSMOTE,SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.metrics import geometric_mean_score

x_train_path=(r'D:\Cooperation\LC_metas_prediction\unbalance_learn_py\FeaSele\ues_PETCT_cohort\LNM\resample_total_data\x_train.csv')
y_train_path=(r'D:\Cooperation\LC_metas_prediction\unbalance_learn_py\FeaSele\ues_PETCT_cohort\LNM\resample_total_data\y_train.csv')
x_train = pd.read_csv(x_train_path, index_col='ID')
y_train = pd.read_csv(y_train_path, index_col='ID')
x_train=x_train.to_numpy()
y_train=y_train.to_numpy().reshape(-1)


import scipy.io as io
# RandomOverSampler
ros=RandomOverSampler(random_state=42)
x_train_ros,y_train_ros=ros.fit_resample(x_train,y_train)
y_train_ros=np.array(y_train_ros,dtype=np.float64)
y_train_ros=y_train_ros.reshape(-1, 1)
io.savemat('x_train_ros.mat',{'x_train':x_train_ros})
io.savemat('y_train_ros.mat',{'y_train':y_train_ros})

### Over-Sampling methods
# ADASYN
ada=ADASYN(random_state=42)
x_train_ada,y_train_ada=ada.fit_resample(x_train,y_train)
y_train_ada=np.array(y_train_ada,dtype=np.float64)
y_train_ada=y_train_ada.reshape(-1, 1)
io.savemat('x_train_ada.mat',{'x_train':x_train_ada})
io.savemat('y_train_ada.mat',{'y_train':y_train_ada})

# SMOTE

x_train_smote,y_train_smote=SMOTE(random_state=42).fit_resample(x_train,y_train)
y_train_smote=np.array(y_train_smote,dtype=np.float64)
y_train_smote=y_train_smote.reshape(-1, 1)
io.savemat('x_train_smote.mat',{'x_train':x_train_smote})
io.savemat('y_train_smote.mat',{'y_train':y_train_smote})

# BorderlineSMOTE
x_train_BorderlineSMOTE,y_train_BorderlineSMOTE=BorderlineSMOTE(random_state=42).fit_resample(x_train,y_train)
y_train_BorderlineSMOTE=np.array(y_train_BorderlineSMOTE,dtype=np.float64)
y_train_BorderlineSMOTE=y_train_BorderlineSMOTE.reshape(-1,1)
io.savemat('x_train_BorderlineSMOTE.mat',{'x_train':x_train_BorderlineSMOTE})
io.savemat('y_train_BorderlineSMOTE.mat',{'y_train':y_train_BorderlineSMOTE})

## KMeansSMOTE
#x_train_KMeansSMOTE,y_train_KMeansSMOTE=KMeansSMOTE().fit_resample(x_train,y_train)     # KMeansSMOTE(kmeans_estimator='KMeans()') 改成这个？原本的设置时适用于大样本
#y_train_KMeansSMOTE=np.array(y_train_KMeansSMOTE,dtype=np.float64)
#y_train_KMeansSMOTE=y_train_KMeansSMOTE.reshape(-1,1)
#io.savemat('x_train_KMeansSMOTE.mat',{'x_train':x_train_KMeansSMOTE})
#io.savemat('y_train_KMeansSMOTE.mat',{'y_train':y_train_KMeansSMOTE})

## SMOTENC   可以用于分类变量

## SVMSMOTE
#x_train_SVMSMOTE,y_trian_SVMSMOTE=SVMSMOTE(random_state=42).fit_resample(x_train,y_train)
#y_trian_SVMSMOTE=np.array(y_trian_SVMSMOTE,dtype=np.float64)
#y_trian_SVMSMOTE=y_trian_SVMSMOTE.reshape(-1,1)
#io.savemat('x_train_SVMSMOTE.mat',{'x_train':x_train_SVMSMOTE})
#io.savemat('y_trian_SVMSMOTE.mat',{'y_train':y_trian_SVMSMOTE})

### Under-sampling methods
# RandomUnderSampler
x_train_rus,y_train_rus=RandomUnderSampler(random_state=42).fit_resample(x_train,y_train)
y_train_rus=np.array(y_train_rus,dtype=np.float64)
y_train_rus=y_train_rus.reshape(-1,1)
io.savemat('x_train_rus.mat',{'x_train':x_train_rus})
io.savemat('y_train_rus.mat',{'y_train':y_train_rus})

#  NearMiss
x_train_NearMiss,y_train_NearMiss=NearMiss(random_state=42).fit_resample(x_train,y_train)
y_train_NearMiss=np.array(y_train_NearMiss,dtype=np.float64)
y_train_NearMiss=y_train_NearMiss.reshape(-1,1)
io.savemat('x_train_NearMiss.mat',{'x_train':x_train_NearMiss})
io.savemat('y_train_NearMiss.mat',{'y_train':y_train_NearMiss})

# TomekLinks
x_train_TomekLinks,y_train_TomekLinks=TomekLinks(random_state=42).fit_resample(x_train,y_train)
y_train_TomekLinks=np.array(y_train_TomekLinks,dtype=np.float64)
y_train_TomekLinks=y_train_TomekLinks.reshape(-1,1)
io.savemat('x_train_TomekLinks.mat',{'x_train':x_train_TomekLinks})
io.savemat('y_train_TomekLinks.mat',{'y_train':y_train_TomekLinks})

# EditedNearestNeighbours
x_train_EditedNearestNeighbours,y_train_EditedNearestNeighbours=EditedNearestNeighbours(random_state=42).fit_resample(x_train,y_train)
y_train_EditedNearestNeighbours=np.array(y_train_EditedNearestNeighbours,dtype=np.float64)
y_train_EditedNearestNeighbours=y_train_EditedNearestNeighbours.reshape(-1,1)
io.savemat('x_train_EditedNearestNeighbours.mat',{'x_train':x_train_EditedNearestNeighbours})
io.savemat('y_train_EditedNearestNeighbours.mat',{'y_train':y_train_EditedNearestNeighbours})

## Combine
# SMOTETomek
x_train_SMOTETomek,y_train_SMOTETomek=SMOTETomek(random_state=42).fit_resample(x_train,y_train)
y_train_SMOTETomek=np.array(y_train_SMOTETomek,dtype=np.float64)
y_train_SMOTETomek=y_train_SMOTETomek.reshape(-1,1)
io.savemat('x_train_SMOTETomek.mat',{'x_train':x_train_SMOTETomek})
io.savemat('y_train_SMOTETomek.mat',{'y_train':y_train_SMOTETomek})

# SMOTEENN
x_train_SMOTEENN,y_train_SMOTEENN=SMOTEENN(random_state=42).fit_resample(x_train,y_train)
y_train_SMOTEENN=np.array(y_train_SMOTEENN,dtype=np.float64)
y_train_SMOTEENN=y_train_SMOTEENN.reshape(-1,1)
io.savemat('x_train_SMOTEENN.mat',{'x_train':x_train_SMOTEENN})
io.savemat('y_train_SMOTEENN.mat',{'y_train':y_train_SMOTEENN})

