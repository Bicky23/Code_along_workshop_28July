# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:57:25 2018

@author: 20120556
"""

import pandas as pd
import numpy as np

#reading the dataset
df=pd.read_csv("E:/training/greyatom/train.csv")
df = df.dropna()
df['Gender'].fillna('Male', inplace=True)


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=0)

x_train=train.drop('Loan_Status',axis=1)
y_train=train['Loan_Status']

x_test=test.drop('Loan_Status',axis=1)
y_test=test['Loan_Status']

#create dummies
x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)
from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
confusion_matrix(y_test, y_pred)

from sklearn.neighbors import KNeighborsClassifier
model = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
model.fit(x_train, y_train)
model.score(x_test,y_test)

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=10,random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)

import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)


