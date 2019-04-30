# Импортируем библиотеки
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
#
# # Создаем датафрейм
# data = pd.read_csv(
#     "data/bank.csv")
#
# from sklearn.preprocessing import LabelEncoder
#
# labelencoder = LabelEncoder()
# for col in data.columns:
#     data[col] = labelencoder.fit_transform(data[col])
#
# # Описываем модель
# model = KMeans(n_clusters=2)
#
# # Проводим моделирование
# model.fit(data.values)
#
# # Предсказание на единичном примере
# predicted_label = model.predict([[5, 5435, 289, 308, 0, 0, 21, 1, 0, 1, 0, 0, 4704, 0]])
#
# # Предсказание на всем наборе данных
# all_predictions = model.predict(data.values)
#
# # Выводим предсказания
# print(predicted_label)
# print(all_predictions)

# Импортируем библиотеки
# from scipy.cluster.hierarchy import linkage, dendrogram
# import matplotlib.pyplot as plt
# import pandas as pd
# 
# # Создаем датафрейм
# data = pd.read_csv(
#     "data/bank.csv")
# 
# from sklearn.preprocessing import LabelEncoder
# 
# # Исключаем информацию об образцах зерна, сохраняем для дальнейшего использования
# varieties = list(data.pop('Exited'))
# 
# labelencoder = LabelEncoder()
# for col in data.columns:
#     data[col] = labelencoder.fit_transform(data[col])
# 
# # Извлекаем измерения как массив NumPy
# samples = data.values
# 
# # Реализация иерархической кластеризации при помощи функции linkage
# mergings = linkage(samples, method='complete')
# 
# # Строим дендрограмму, указав параметры удобные для отображения
# dendrogram(mergings,
#            labels=varieties,
#            leaf_rotation=90,
#            leaf_font_size=6,
#            )
# 
# plt.show()

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# read data (drop last empty column, caused by an extra (last) colon in the header)
data = pd.read_csv("data/bank.csv", sep=',').dropna(axis=1, how='all')

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

# normalize data
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('RowNumber', 1))

# clustering
n_clusters = 2
km = KMeans(n_clusters=n_clusters)

# fit & predict clusters
data['cluster'] = km.fit_predict(X)

# results - we should have 2 clusters: [0,1]
print(data)

# cluster's centroids
print(km.cluster_centers_)

test = pd.read_csv("data/bank_test.csv", sep=',').dropna(axis=1, how='all')

for col in test.columns:
    test[col] = labelencoder.fit_transform(test[col])

X_new = scaler.transform(test.drop('RowNumber', 1))


test['cluster'] = km.predict(X_new)

# results - we should have 2 clusters: [0,1]
print(test)

param1 = 6
param2 = 1

# Разделение набора данных
x_axis = data.values[:, param1]  # Sepal Length
y_axis = data.values[:, param2]  # Sepal Width

print("X_axis:", x_axis)
print("Y_axis:",y_axis)

# Построение
plt.xlabel(data.columns[param1])
plt.ylabel(data.columns[param2])
plt.scatter(x_axis, y_axis, c=data["cluster"])
plt.show()