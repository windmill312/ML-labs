from subprocess import check_output
print(check_output(["ls", "data"]).decode("utf8"))

import pandas as pd
import numpy as np

data = pd.read_csv("data/bank.csv")
print(data.groupby('Exited').size())
print(data.head(6))

#Пустые значения
print(data.isnull().sum())

data['Exited'].unique()

#Размерность таблицы
print(data.shape)

#Замена строковых типов числовым
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

print(data.head())

#Проверка замены типов
print(data['Geography'].unique())

#??
X = data.iloc[:,1:]
y = data.iloc[:,13]
print(X.head())
print(y.head())

print(X.describe())

print(data.corr())
print("-----------------------------------")
print("Логистическая регрессия")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model_LR = LogisticRegression()

model_LR.fit(X_train,y_train)

y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(confusion_matrix)

auc_roc = metrics.roc_auc_score(y_test,y_pred)
print("ROC_AUC:")
print(auc_roc)

LR_ridge = LogisticRegression()
LR_ridge.fit(X_train,y_train)

rep = metrics.classification_report(y_test,y_pred)
print(rep)
print("-----------------------------------")
print("Случайный лес")
from sklearn.ensemble import RandomForestClassifier

model_RR=RandomForestClassifier()

model_RR.fit(X_train,y_train)
#%%
y_prob = model_RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(confusion_matrix)

auc_roc=metrics.roc_auc_score(y_test,y_pred)
print("ROC_AUC:")
print(auc_roc)

rep=metrics.classification_report(y_test,y_pred)
print(rep)
print("-----------------------------------")
print("К-случайных соседей")
from sklearn.neighbors import KNeighborsClassifier

model_KN=KNeighborsClassifier()

model_KN.fit(X_train, y_train)

y_prob = model_RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(confusion_matrix)

rep=metrics.classification_report(y_test,y_pred)
print(rep)

auc_roc=metrics.roc_auc_score(y_test,y_pred)
print("ROC_AUC:")
print(auc_roc)