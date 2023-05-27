import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Iris.csv',sep=',')
data

data = data.drop(['Id'],axis=1)
data

data.Species.value_counts()

le = LabelEncoder()

data['Species'] = le.fit_transform(data['Species'])

data

# Setosa - 0
# versicolor - 1
# virginica - 2



x = data.drop(['Species'],axis =1)
x
y = data['Species']
y

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_pred = lr.predict(x_test)
lr_score = accuracy_score(y_test,lr_pred)
lr_score


# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()
# dt.fit(x_train,y_train)
# y_pred = dt.predict(x_test)
# dt_score = accuracy_score(y_test,y_pred)
# dt_score

pickle.dump(lr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

model.predict([[3.8,3.2,1.3,0.3]])


