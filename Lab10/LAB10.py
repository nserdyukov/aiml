import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Shopping_data.csv')
dataset.head()
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 3].values
y *= 1000
print ("Матрица признаков"); print(X[:5])
print ("Зависимая переменная"); print(y[:5])
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
from sklearn.preprocessing import OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
from sklearn.linear_model import LinearRegression
X = X.reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# прогнозирование значения зависимой переменной
# с использованием линейной и полиноминальной модели для значения 6.5.
y_pred_lin = lin_reg.predict([[6.5]])
y_pred_poly = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(y_pred_lin, y_pred_poly)

# точечная форма изображения регрессии
plt.scatter(X, y, color = 'red')

# построение линейной регрессии
plt.plot(X, lin_reg.predict(X), color = 'blue')

plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Annual Income')
plt.show()
# Создание массива значений X для более плавной линии регрессии
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

# точечная форма изображения регрессии
plt.scatter(X, y, color = 'red')

# построение линейной регрессии
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')

plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Annual Income')
plt.show()