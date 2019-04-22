# In[1]:
import pandas as pd
import numpy as np
import random

SEED = 1408
random.seed(SEED)
np.random.seed(SEED)

train = pd.read_csv('data/google.csv')

print('Train shape:', train.shape)

train.info()

train.head()

train.corr()

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


y = train['Rating']
X = train.drop('Rating', axis=1)

print(X.shape, y.shape)

# In[11]:
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=8)
# In[12]:
print(X_train.shape, X_valid.shape, y_train, y_valid)

from sklearn.tree import DecisionTreeClassifier
first_tree = DecisionTreeClassifier(random_state=8)
np.mean(cross_val_score(first_tree, X_train, y_train, cv=5))

from sklearn.neighbors import KNeighborsClassifier
first_knn = KNeighborsClassifier()
np.mean(cross_val_score(first_knn, X_train, y_train, cv=5))

# clf = LogisticRegression(max_iter=100, solver='lbfgs', n_jobs=-1, multi_class='auto')
# clf.fit(X, y)
# print('Confusion matrix:', confusion_matrix(y, clf.predict(X)))
# print('Classification report for test data:')
# print(classification_report(y, clf.predict(X)))
#
#
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