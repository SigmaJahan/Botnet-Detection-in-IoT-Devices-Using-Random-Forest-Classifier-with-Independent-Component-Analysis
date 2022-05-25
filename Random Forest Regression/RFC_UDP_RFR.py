#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the benign dataset
dataset_benign = pd.read_csv('benign_traffic.csv', delimiter=',')
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros
attack = zerolistmaker(175240)
dataset_benign['Attack'] = attack
dataset_benign.head()

#Importing the Attack dataset (UDP)
dataset_attack = pd.read_csv('udp.csv', delimiter=',')
def onelistmaker(n):
    listofones = [1] * n
    return listofones
attack = onelistmaker(217034)
dataset_attack['Attack'] = attack
dataset_attack.head()

#Concat
df_udp = pd.concat([dataset_benign, dataset_attack])
X = df_udp.drop('Attack',1)
y = df_udp['Attack']



#RFR
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=0, max_depth=10)
rfr.fit(X,y)

features = df_udp.columns 
importances = rfr.feature_importances_ 
indices = np.argsort(importances)[-14:]  # top 15 features 
plt.title('Feature Importances') 
plt.barh(range(len(indices)), importances[indices], color='b', align='center') 
plt.yticks(range(len(indices)), [features[i] for i in indices]) 
plt.xlabel('Relative Importance') 
plt.show()
df_udp.head()

from sklearn.feature_selection import SelectFromModel 
feature = SelectFromModel(rfr,threshold=0.05)
Fit = feature.fit_transform(X,y)
X=Fit

#Splitting the test train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Normalizing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the Test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Scatter Plot with Hue for visualizing data in 3-D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as sns
cols = ['H_L1_weight', 'MI_dir_L0.1_weight', 'H_L0.01_weight', 'H_L0.1_weight', 'MI_dir_L1_weight','Attack']
pp = sns.pairplot(df_udp[cols], hue='Attack', size=1.8, aspect=1.8, 
                  palette={1: "#FF9999", 0: "#FFE888"},
                  plot_kws=dict(edgecolor="black", linewidth=0.5))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
