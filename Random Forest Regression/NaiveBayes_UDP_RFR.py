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
feature = SelectFromModel(rfr,threshold=0.25)
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
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#Predicting the Test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#ROC generation
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

probs=classifier.predict_proba(X_test)
probs=probs[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)





