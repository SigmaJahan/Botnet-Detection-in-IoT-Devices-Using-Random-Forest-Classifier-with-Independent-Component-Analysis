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

#Splitting the test train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Normalizing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test= pd.DataFrame(sc.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

#High Correlation Filter threshold taken=0.5
correlated_features = set()
correlation_matrix = X.corr()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
print(correlated_features)
print(len(correlated_features)) 


X_train.drop(labels=correlated_features, axis=1, inplace=True)
X_test.drop(labels=correlated_features, axis=1, inplace=True)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the Test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




