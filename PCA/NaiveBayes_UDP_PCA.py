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

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Normalizing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

plt.plot(range(2), pca.explained_variance_ratio_)
plt.plot(range(2), np.cumsum(pca.explained_variance_ratio_))
plt.title("Component-wise and Cumulative Explained Variance")

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#Predicting the Test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results



