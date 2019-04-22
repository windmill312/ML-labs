# In[1]:
import pandas as pd
import numpy as np
import random

SEED = 1408
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('data/chocolate.csv')

print('Train shape:', train.shape)
train.head()

train.corr()

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
#
# from sklearn.metrics import classification_report, confusion_matrix
#
# X = train.loc[:, train.columns != 'Rating']
# y = train['Rating']
#
# #print(X, y)
#
# clf = LogisticRegression(max_iter=100, solver='lbfgs', n_jobs=-1, multi_class='auto')
# clf.fit(X, y)
# print('Confusion matrix:', confusion_matrix(y, clf.predict(X)))
# print('Classification report for test data:')
# print(classification_report(y, clf.predict(X)))


#
#
# # In[11]:
#
#
# coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(clf.coef_))], axis = 1)
# coefficients.columns = ['feature', 'Rating1', 'Rating2', 'Rating3']
# print(coefficients)
#
#
# # In[12]:
#
#
# clf = KNeighborsClassifier(n_jobs=-1)
# clf.fit(X, y)
# print('Confusion matrix:', confusion_matrix(y, clf.predict(X)))
# print('Classification report for test data:')
# print(classification_report(y, clf.predict(X)))
#
#
# # In[13]:
#
#
# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# clf.fit(X, y)
# print('Confusion matrix:', confusion_matrix(y, clf.predict(X)))
# print('Classification report for test data:')
# print(classification_report(y, clf.predict(X)))
#
#
# In[ ]: