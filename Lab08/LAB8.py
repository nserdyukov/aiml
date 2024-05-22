import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
dataset = pd.read_csv('placement-dataset.csv')
dataset.head()
X = dataset.iloc[:, 1:-2].values
y = dataset.iloc[:, 2].values
print ("Матрица признаков"); print(X[:5])
print ("Зависимая переменная"); print(y[:5])
from sklearn.impute import SimpleImputer
import numpy as np

# Предполагая, что X - ваш набор данных
# Измените форму X, чтобы убедиться, что это 2D массив
X = X.reshape(-1, 1)

# Создание экземпляра SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Обучение импутера на данных
X = imputer.fit_transform(X)

# Предполагая, что y - другой набор данных
# Измените форму y, чтобы убедиться, что это 2D массив
y = y.reshape(-1, 1)

# Создание отдельного экземпляра SimpleImputer для y
imputer2 = SimpleImputer(missing_values=np.nan, strategy='mean')

# Обучение импутера на данных
y = imputer2.fit_transform(y)

# Вывод обновленных наборов данных
print(X)
print(y)
# Сортировка массива X
X = np.sort(X, axis=0)

# Сортировка массива y
y = np.sort(y, axis=0)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
# Отображание точкек данных обучающего набора
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Отображание точкек данных тестового набора
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()