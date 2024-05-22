from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

data_source = 'Dataset.txt'
d = pd.read_table(data_source, delimiter=',',
                  header=None,
                  names=['id','Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','EquivDiameter','Extent','Perimeter','Roundness','AspectRation','Class'])
d.head()
X_train = d[['id','Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','EquivDiameter','Extent','Perimeter','Roundness','AspectRation']]
y_train = d['Class']


K = 3

# Создание и настройка классификатора
knn = KNeighborsClassifier(n_neighbors=K)
# построение модели классификатора (процедура обучения)
knn.fit(X_train.values, y_train)

# Использование классификатора
# Объявление признаков объекта
X_test = np.array([[10512,4802,145.8326338,42.78482354,0.9559949704,4929,78.19268696,0.4200122453,318.762,0.5938803197,3.408513154]])
# Получение ответа для нового объекта
target = knn.predict(X_test)
print(target)